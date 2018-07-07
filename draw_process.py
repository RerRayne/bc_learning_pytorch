import os


def draw_progress(train_loss, val_loss,
                  train_error_rate, val_error_rate,
                  curr_epoch,
                  cv, dir_to_save):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, "r", label='Train loss')
    plt.plot(val_loss, "g", label='Val loss')
    plt.title("CV: {}, Epoch {}".format(cv, curr_epoch))
    plt.xlabel("epoch")
    plt.ylabel("KL div")
    plt.xlim(0, len(train_loss))
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_error_rate, "r", label='Train error_rate')
    plt.plot(val_error_rate, "g", label='Test error_rate')
    plt.title("CV: {}, Epoch {}".format(cv, curr_epoch))
    plt.xlabel("epoch")
    plt.ylabel("error rate")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(dir_to_save, "progress_cv{}.png".format(cv)))
