import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
#import numpy as np

#classes = ['fire', 'smoke']
cm = confusion_matrix(truelabels, predictions)
#tick_marks = np.arange(len(classes))
df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (20,10))
annot_kws={'fontsize':14,
           'color':"k",
           'verticalalignment':'center'}

sns.heatmap(df_cm, annot = True, annot_kws = annot_kws, cmap ='Blues', vmin = 0, vmax = 100, linewidth = 0.2, linecolor = 'k')
plt.xlabel('Predicted shape', fontsize = 24)
plt.ylabel('True shape', fontsize = 24)
plt.show()

print(classification_report(truelabels, predictions))


plt.figure(1, figsize = (15,5))
# Plot loss
plt.subplot(1,2,1)
plt.plot(train_losses, '-bx', label = 'Training loss')
plt.plot(valid_losses, '-rx', label = 'Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')
plt.grid()
plt.legend(frameon = False)

# Plot accuracy
plt.subplot(1,2,2)
plt.plot(train_acc, '-bx', label = 'Training acc')
plt.plot(valid_acc, '-rx', label = 'Validation acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Accuracy vs. No. of epochs')
plt.legend(frameon = False)
plt.grid()
plt.show()
