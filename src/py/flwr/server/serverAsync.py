"""Flower asynchronous server."""


import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import List, Optional, Tuple, cast

import threading
import time

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Reconnect,
    Weights,
    parameters_to_weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import DefaultStrategy, Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]

# Implemenetation details:
# - How serverAsync knows it should start another round of training? How long it 
# should wait before starting next round?


class ServerAsync:
    "Flower asynchronous server"

    def __init__(
        self, 
        client_manager: ClientManager = None, 
        strategy: Optional[Strategy] = None,
        round_timeout: int = 30,
    ) -> None:
        self._client_manager: ClientManager = client_manager if client_manager is not None else SimpleClientManager()
        self.weights: Weights = []
        self.strategy: Strategy = strategy if strategy is not None else DefaultStrategy()
        self.round_timeout = round_timeout
        self.lock = threading.Lock()

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def fit(self, num_rounds: int) -> History:
        """Run ASYNC federated averaging for a number of rounds."""
        history = History()
        # Initialize weights by asking one client to return theirs
        self.weights = self._get_initial_weights()

        # Evaluate the current model before training.
        # This pre-evaluation might not be done, depending on whether the 
        # startegy is provided with an eval_fn. Check cifar_app/server.py.
        res = self.strategy.evaluate(weights=self.weights)
        if res is not None:
            log(
                INFO,
                "initial weights (loss/accuracy): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_accuracy_centralized(rnd=0, acc=res[1])

        # Run federated learning for num_rounds
        log(INFO, "[TIME] FL starting")
        start_time = timeit.default_timer()

        # For now, assume that num_rounds = 1
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            weights_prime = self.start_fit_round(rnd=current_round)
            time.sleep(self.round_timeout)

            with self.lock:
                # Evaluate model using strategy implementation
                res_cen = self.strategy.evaluate(weights=self.weights)
                if res_cen is not None:
                    loss_cen, acc_cen = res_cen
                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        acc_cen,
                        timeit.default_timer() - start_time,
                    )
                    history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                    history.add_accuracy_centralized(rnd=current_round, acc=acc_cen)

                # Evaluate model on a sample of available clients
                res_fed = self.evaluate(rnd=current_round)
                if res_fed is not None and res_fed[0] is not None:
                    loss_fed, _ = res_fed
                    history.add_loss_distributed(
                        rnd=current_round, loss=cast(float, loss_fed)
                    )

                # Conclude round
                loss = res_cen[0] if res_cen is not None else None
                acc = res_cen[1] if res_cen is not None else None
                should_continue = self.strategy.on_conclude_round(current_round, loss, acc)
                if not should_continue:
                    break

        # Send shutdown signal to all clients
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "[TIME] FL finished in %s", elapsed)
        return history

    def start_fit_round(self, rnd: int) -> Optional[Weights]:    
        """Start one round of ASYNC fitting"""

        # Get clients and their respective instructions from strategy
        # client_instruction: List[Tuple[ClientProxy, FitIns]]
        client_instructions = self.strategy.on_configure_fit(
            rnd=rnd, weights=self.weights, client_manager=self._client_manager
        )
        log(
            DEBUG,
            "start_fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )
        if not client_instructions:
            log(INFO, "start_fit_round: no clients sampled, cancel fit")
            return None

        # Send instructions to each client
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for c, ins in client_instructions:
                executor.submit(self.fit_client_and_update_model, c, ins, rnd)

    def fit_client_and_update_model(
        self, 
        client: ClientProxy, 
        ins: FitIns,
        rnd: int,
    ) -> Tuple[ClientProxy, FitRes]:
        """
        Refine weights on a single client, and update server's model immediately 
        after receiving the computation result.
        """
        fit_res = client.fit(ins)
        weights_prime = self.strategy.on_aggregate_fit(rnd, [(client, fit_res)], [])
        if weights_prime is not None:
            with self.lock:
                log(INFO, "Server receives model update from client {}".format(client.cid))
                self.weights = weights_prime

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.on_configure_evaluate(
            rnd=rnd, weights=self.weights, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate: no clients sampled, cancel federated evaluation")
            return None
        log(
            DEBUG,
            "evaluate: strategy sampled %s clients",
            len(client_instructions),
        )

        # Evaluate current global weights on those clients
        results_and_failures = evaluate_clients(client_instructions)
        results, failures = results_and_failures
        log(
            DEBUG,
            "evaluate received %s results and %s failures",
            len(results),
            len(failures),
        )
        # Aggregate the evaluation results
        loss_aggregated = self.strategy.on_aggregate_evaluate(rnd, results, failures)
        return loss_aggregated, results_and_failures


    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients."""
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        return parameters_to_weights(parameters_res.parameters)



def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures

def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect

def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            results.append(future.result())
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate weights on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
    