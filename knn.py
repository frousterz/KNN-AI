# Autor:     Felipe Salas Casasola
# Carnet:    2014129277
# Curso:     Inteligencia Artificial
# Profesor:  Jose Carranza

# Codepad
# link Collaboration: https://codepad.remoteinterview.io/QRBULOOYTP
# Editor Personal: https://codepad.remoteinterview.io/OXARXJQEKI


import numpy as np
import pickle
import heapq as hq

class Image:
  def __init__(self, data, label):
    self.data = data
    self.label = label

  def get_data(self):
    return self.data

  def get_label(self):
    return self.label


class ImageClassifier:
  def __init__(self, training_data, test_data):
    self.training_data = training_data
    self.test_data = test_data
    self.hits = 0
  
  def manhattan(self, k = 1):
    test = self.test_data
    training = self.training_data
    #len(test)
    for test_iterator in range(10):
      results = []
      test_element = test[test_iterator].data

      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        results.append(np.sum(abs(test_element - training_element)))

      min_distances = hq.nsmallest(k, results)
      for iterator in range(len(min_distances)):
        min_index = results.index(min_distances[iterator])
        if(self.training_data[min_index].label == self.test_data[test_iterator].label):
          self.hits += 1
          break
            
      print("Iteracion ->" + str(test_iterator))
          
    print("k = " + str(k)
          + " :: aciertos: " + str(self.hits)
          + " :: % exito: " + str((self.hits*100)/10))

  def chebyshev(self, k = 1):
    test = self.test_data
    training = self.training_data
    for test_iterator in range(10):
      results = []
      test_element = test[test_iterator].data

      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        results.append(max(np.absolute(test_element - training_element)))

      if(k == 1):
        max_index = np.argmax(results)
        if(self.training_data[max_index].label == self.test_data[test_iterator].label):
          self.hits += 1

      if(k == 2):
        max_distances = hq.nlargest(2, results)
        for iterator in range(len(max_distances)):
          max_index = results.index(max_distances[iterator])
          if(self.training_data[max_index].label == self.test_data[test_iterator].label):
            self.hits += 1

      if(k == 3):
        max_distances = hq.nlargest(3, results)
        for iterator in range(len(max_distances)):
          max_index = results.index(max_distances[iterator])
          if(self.training_data[max_index].label == self.test_data[test_iterator].label):
            self.hits += 1
            
      print("Iteracion ->" + str(test_iterator))
          
    print("k = " + str(k)
          + " :: aciertos: " + str(self.hits)
          + " :: % exito: " + str((self.hits*100)/10))
    
  def levenshtein(self, k = 1):
    test = self.test_data
    training = self.training_data
    # len(test)
    for test_iterator in range(100):
      results = []
      test_element = test[test_iterator].data

      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        diff = np.where(test_element != training_element)
        results.append(len(diff[0]))

      if(k == 1):
        min_index = np.argmin(results)
        if(self.training_data[min_index].label == self.test_data[test_iterator].label):
          self.hits += 1

      if(k == 2):
        min_changes = hq.nsmallest(2, results)
        for iterator in range(len(min_changes)):
          min_index = results.index(min_changes[iterator])
          if(self.training_data[min_index].label == self.test_data[test_iterator].label):
            self.hits += 1
            break

      if(k == 3):
        min_changes = hq.nsmallest(3, results)
        for iterator in range(len(min_changes)):
          min_index = results.index(min_changes[iterator])
          if(self.training_data[min_index].label == self.test_data[test_iterator].label):
            self.hits += 1
            break
      
      print("Iteracion ->" + str(test_iterator))
          
    print("k = " + str(k)
          + " :: aciertos: " + str(self.hits)
          + " :: % exito: " + str((self.hits*100)/100))

    
class DataManager:
  def __init__(self):
    self.training_data = []
    self.test_data = []

  def get_training_data(self):
    return self.training_data
  
  def get_test_data(self):
    return self.test_data

  def load_training_data(self):
    print("Loading training data...")
    for n in range(1,2):
      print("File Loaded!")
      with open("test/data_batch_" + str(n), 'rb') as file_path:
        file_loaded = pickle.load(file_path, encoding='bytes')
        data_loaded = file_loaded[b'data']
        labels_loaded = file_loaded[b'labels']
        for it in range (len(data_loaded)):
          self.training_data.append(Image(data_loaded[it], labels_loaded[it]))
    print("Training data loaded successfully!")

  def load_test_data(self):
    print("Loading test data...")
    with open("test/test_batch", 'rb') as file_path:
      file_loaded = pickle.load(file_path, encoding='bytes')
      data_loaded = file_loaded[b'data']
      labels_loaded = file_loaded[b'labels']
      for it in range (len(data_loaded)):
        self.test_data.append(Image(data_loaded[it], labels_loaded[it]))
    print("Test data loaded successfully!")


"""
  The main function is used to create and call objects and actions/methods
  to test all the functionality.
"""

def main():
  print("Hello KNN")

dm = DataManager()
dm.load_training_data()
dm.load_test_data()
i = ImageClassifier(dm.training_data, dm.test_data)

"""
Para probar todo solo use:
>>> i.distance_method()
ahi escoge el que quiera.
por defecto k =1 asi que no es necesario que le ponga parametro, a no ser de que use k = 2 o k=3

Si quiere probar otro vuelva a reiniciar el estos pasos pero con otro metodo de distancia.
"""

