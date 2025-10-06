import os

def world_info():
    # Single worker by default in your setup
    return {
        'rank': int(os.environ.get('RANK', '0')),
        'world_size': int(os.environ.get('WORLD_SIZE', '1')),
        'local_rank': int(os.environ.get('LOCAL_RANK', '0')),
    }
