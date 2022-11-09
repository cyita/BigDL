from bigdl.orca.common import init_orca_context

sc = init_orca_context(cluster_mode="standalone", num_nodes=4, enable_numa_binding=True)