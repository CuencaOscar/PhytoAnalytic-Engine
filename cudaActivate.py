import torch

cuda_available = torch.cuda.is_available()

print("CUDA disponible:", cuda_available)

# Si CUDA está disponible, cantidad de GPUs y el nombre de la primera
if cuda_available:
    print("Cantidad de GPUs disponibles:", torch.cuda.device_count())
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
