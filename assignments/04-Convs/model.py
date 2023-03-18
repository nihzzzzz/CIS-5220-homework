import torch


class Model(torch.nn.Module):
    """
    Define the CNN Model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Arguments:
            num_channels: int, number of input channels
            num_classes: int, number of output classes
        Returns:
            None
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(
            num_channels, 16, kernel_size=3, stride=2, padding=1
        )
        self.batch1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        self.fc1 = torch.nn.Linear(16 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            CNN x: torch.Tensor
        Returns:
            CNN model x: torch.Tensor
        """

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
