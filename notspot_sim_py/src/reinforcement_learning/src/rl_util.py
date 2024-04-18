from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
import torch.optim as optim

def ros_image_to_pytorch_tensor(image_msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to a NumPy array
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    
    # Assuming cv_image is a numpy.uint16 array from a depth image
    # Convert the numpy array to float32
    cv_image_float = cv_image.astype(np.float32)
    
    # Convert the NumPy array (now float32) to a PyTorch tensor
    tensor_image = torch.from_numpy(cv_image_float)
    
    return tensor_image



def set_optimizer(model, hyperparameters):
    optimizer_name = hyperparameters['optimizer']
    learning_rate = hyperparameters['lr']

    # Dictionary mapping optimizer names to optimizer classes
    optimizer_mapping = {
        'Adam': optim.Adam,
        'Adadelta': optim.Adadelta,
        'Adagrad': optim.Adagrad,
        'Adamax': optim.Adamax,
        'ASGD': optim.ASGD,
        'LBFGS': optim.LBFGS,
        'RMSprop': optim.RMSprop,
        'Rprop': optim.Rprop,
        'SGD': optim.SGD,
    }

    if optimizer_name in optimizer_mapping:
        optimizer_class = optimizer_mapping[optimizer_name]
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    return optimizer


def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
    except:
        pass