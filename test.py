import os; os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from lib.attention import test_forward_attention

def main():
    test_forward_attention()

if __name__ == '__main__':
    main()
