import torch


def to_device(data, device):
    """Mover tensor(es) al dispositivo elegido"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_default_device():
    """Seleccionar GPU si está disponible, de lo contrario, CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class DeviceDataLoader():
    """Envolver un DataLoader para mover datos a un dispositivo"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Proporcionar un lote de datos después de moverlo al dispositivo"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Número de lotes"""
        return len(self.dl)
