
import os
import torch




def save_model(model, device, model_dir='models/dicts', model_file_name='classifier.pt'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    
    if device == 'cuda':
        model.to('cuda')
    
    return


def save_model_2(model, device, model_dir='models/dicts', model_file_name='classifier.pt'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        torch.save(model.module.state_dict(), model_path)
    else:
        # save the state_dict
        torch.save(model.state_dict(), model_path)

    return


def save_model_3(model_name, model, optimizer):
    model_file = 'checkpoints/' + model_name + '.pth'
    torch.save({
                'model_state_dict': model,
                'optimizer_state_dict': optimizer,
                }, 
                model_file)

