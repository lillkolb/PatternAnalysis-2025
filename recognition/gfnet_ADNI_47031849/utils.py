"""
From Task Sheet:
"You may create other helper files such as “utils.py” to better organise your project"
"""

import matplotlib.pyplot as plt
def plot_data():
    # load accuracy and loss data
    with open(os.path.join(saving_filepath, 'train_loss_data.pkl'), 'rb') as f:
        tld_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'train_accuracy_data.pkl'), 'rb') as f:
        tad_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'valid_loss_data.pkl'), 'rb') as f:
        vld_load = pickle.load(f)

    with open(os.path.join(saving_filepath, 'valid_accuracy_data.pkl'), 'rb') as f:
        vad_load = pickle.load(f)

    total_epochs = len(tld_load)
    ep = np.arange(0, total_epochs, 1)
    
    # plt.subplot(1, 2, 1)
    plt.plot(ep, tld_load, 'g', ep, vld_load, 'r', linewidth=2.0)
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid()
    plt.show()

    # plt.subplot(1, 2, 2)
    plt.plot(ep, tad_load, 'g', ep, vad_load, 'r', linewidth=2.0)
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.grid()
    plt.show()

