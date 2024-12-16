import torch
from torch.utils.tensorboard import SummaryWriter

from src.models.efficientnet_b0 import get_model as get_efficientnet_b0
from src.models.mobilenet_v3_large import get_model as get_mobilenet_v3_large
from src.models.resnet50 import get_model as get_resnet50


writer = {
    'model1': SummaryWriter("logs/efficientnet_b0"),
    'model2': SummaryWriter("logs/mobilenet_v3_large"),
    'model3': SummaryWriter("logs/resnet50")
}
# writer = SummaryWriter("logs")
model1 = get_efficientnet_b0(120)
model2 = get_mobilenet_v3_large(120)
model3 = get_resnet50(120)
writer['model1'].add_graph(model1, input_to_model=torch.randn(1, 3, 224, 224))
writer['model2'].add_graph(model2, input_to_model=torch.randn(1, 3, 224, 224))
writer['model3'].add_graph(model3, input_to_model=torch.randn(1, 3, 224, 224))
# writer.add_graph(model3, input_to_model=torch.randn(1,3,244,244))
writer['model1'].close()
writer['model2'].close()
writer['model3'].close()
# writer.close()

print("OK")