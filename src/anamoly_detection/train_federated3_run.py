import subprocess

subprocess.Popen('python3 train_federated3.py --job_name "ps" --task_index 0', shell = True)
subprocess.Popen('python3 train_federated3.py --job_name "worker" --task_index 0', shell = True)
subprocess.Popen('python3 train_federated3.py --job_name "worker" --task_index 1', shell = True)
subprocess.Popen('python3 train_federated3.py --job_name "worker" --task_index 2', shell = True)
#subprocess.Popen('python3 syn_distributed_tf_movielens4.py --job_name "worker" --task_index 3', shell = True)
#subprocess.Popen('python3 syn_distributed_tf_movielens4.py --job_name "worker" --task_index 4', shell = True)
#subprocess.Popen('python3 syn_distributed_tf_movielens4.py --job_name "worker" --task_index 5', shell = True)