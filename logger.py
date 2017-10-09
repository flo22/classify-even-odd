"""
Logger for logging model and results as confusion matrix
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model


class Logger:

    def __init__(self):
        """
        Constructor
        separates logging config and model from confusion matrix
        """
        print("start Logger")
        self.log_path = './data/logs/'

        # log_model arguments
        self.x = 0
        self.path = self.log_path + 'config%s.txt' % self.x
        self.model_path = self.log_path + 'model%s.png' % self.x
        self.summary_path = self.log_path + 'summary%s.txt' % self.x
        self.hdf5_path = self.log_path + 'model%s.hdf5' % self.x
        # log_confusion arguments
        self.y = 0
        self.cm_path = self.log_path + 'cm%s.png' % self.y
        self.cmn_path = self.log_path + 'cmn%s.png' % self.y
        self.confusion_precision = 2

    # define the confusion matrix function
    @staticmethod
    def plot_confusion_matrix(cm,
                              classes,
                              title='Confusion Matrix',
                              normalize=False,
                              cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion Matrix")
        else:
            print('Confusion Matrix')
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # log and plot model
    def log_model(self,
                  model,
                  log_config=True,
                  log_summary=True,
                  log_model=True,
                  save_model=False):
        while os.path.exists(self.path):
            self.x += 1
            self.path = self.log_path + 'config%s.txt' % self.x
            self.summary_path = self.log_path + 'summary%s.txt' % self.x
            self.model_path = self.log_path + 'model%s.png' % self.x
            self.hdf5_path = self.log_path + 'model%s.hdf5' % self.x
        # write model config into .txt
        if log_config:
            file = open(self.path, 'w+')
            summary = str(model.get_config())
            file.write(summary)
            file.close()
        # write model summary into .txt
        if log_summary:
            with open(self.summary_path, 'w+') as ms:
                model.summary(print_fn=lambda x: ms.write(x + '\n'))
        # plot model as .png
        if log_model:
            print('saved model into ' + self.path)
            plot_model(model, to_file=self.model_path)
        # save the model as .hdf5
        if save_model:
            model.save(self.hdf5_path)

    # log and plot confusion matrix
    def log_confusion(self,
                      labels,
                      predictions,
                      classes):
        cnf_matrix = confusion_matrix(labels, predictions)
        np.set_printoptions(precision=self.confusion_precision)
        fig1 = plt.figure()
        self.plot_confusion_matrix(cnf_matrix,
                                   classes=classes,
                                   title='Confusion Matrix')

        fig2 = plt.figure()
        self.plot_confusion_matrix(cnf_matrix,
                                   classes=classes,
                                   normalize=True,
                                   title='Normalized Confusion Matrix')

        while os.path.exists(self.cm_path):
            self.y += 1
            self.cm_path = self.log_path + 'cm%s.png' % self.y
            self.cmn_path = self.log_path + 'cmn%s.png' % self.y
        fig1.savefig(self.cm_path)
        fig2.savefig(self.cmn_path)
        # plt.show() # show confusion matrix
