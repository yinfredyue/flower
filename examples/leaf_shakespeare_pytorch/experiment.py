"""
This script is not working yet. But you can use the following commands to
start server and clients.
"""
import subprocess


# python server.py --num_clients 3 --rounds 25 --staleness_bound 2  > ssp_s_2_c_3_server.log 2>&1
# python client.py --num_clients 3 --staleness_bound 2 --idx 0 > ssp_s_2_c_3_client0.log 2>&1
# python client.py --num_clients 3 --staleness_bound 2 --idx 1 > ssp_s_2_c_3_client1.log 2>&1
# python client.py --num_clients 3 --staleness_bound 2 --idx 2 > ssp_s_2_c_3_client2.log 2>&1

num_clients = 3
# staleness_list = [2, 3, 4, 5, 6, 7, 10]
staleness_list = [2]

for s in staleness_list:
    # Start server and get its log
    server_log_file = f"ssp_s_{s}_c_{num_clients}_server.log"
    subprocess.run(f"python server.py --num_clients {num_clients} --staleness_bound {s} >{server_log_file} 2>&1 &", shell=True)

    # Start clients and get their log
    for idx in range(0, num_clients):
        client_log_file = f"ssp_s_{s}_c_{num_clients}_client{idx}.log"
        subprocess.run(f"python client.py --num_clients {num_clients} --idx {idx} --staleness_bound {s} >{client_log_file} 2>&1 &", shell=True)




