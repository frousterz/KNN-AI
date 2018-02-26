# Autor:     Felipe Salas Casasola
# Carnet:    2014129277
# Curso:     Inteligencia Artificial
# Profesor:  Jose Carranza
# Tarea: Implementacion del Algoritmo: K Nearest Neighbors
# Fecha:     25/02/2018

import numpy as np
import pickle as pckl
import heapq as hq
from tabulate import tabulate

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
    
  def levenshtein(self, k = 1):
    test = self.test_data
    training = self.training_data
    for test_iterator in range(len(test)):
      results = []
      test_element = test[test_iterator].data
      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        diff = np.where(test_element != training_element)
        results.append(len(diff[0]))

      min_changes = hq.nsmallest(k, results)
      for iterator in range(len(min_changes)):
        min_index = results.index(min_changes[iterator])
        if(self.training_data[min_index].label == self.test_data[test_iterator].label):
          self.hits += 1
          break          
    tabulate_result(k, self.hits, ((self.hits*100))/len(test))

  def manhattan(self, k = 1):
    test = self.test_data
    training = self.training_data
    for test_iterator in range(len(test)):
      results = []
      test_element = test[test_iterator].data

      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        results.append(np.sum(np.abs(test_element - training_element)))

      min_distances = hq.nsmallest(k, results)        
      for iterator in range(len(min_distances)):
        min_index = results.index(min_distances[iterator])
        if(self.training_data[min_index].label == self.test_data[test_iterator].label):
          self.hits += 1
          break
    tabulate_result(k, self.hits, ((self.hits*100))/len(test))

  def chebyshev(self, k = 1):
    test = self.test_data
    training = self.training_data
    for test_iterator in range(len(test)):
      results = []
      test_element = test[test_iterator].data

      for training_iterator in range(len(training)):
        training_element = training[training_iterator].data
        results.append(np.amax(np.absolute(test_element - training_element)))

      min_distances = hq.nsmallest(k, results)
      for iterator in range(len(min_distances)):
        min_index = results.index(min_distances[iterator])
        if(self.training_data[min_index].label == self.test_data[test_iterator].label):
          self.hits += 1
          break
    tabulate_result(k, self.hits, ((self.hits*100))/len(test))


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
    for n in range(2,3):
      print("File Loaded!")
      with open("test/data_batch_" + str(n), 'rb') as file_path:
        file_loaded = pckl.load(file_path, encoding='bytes')
        data_loaded = file_loaded[b'data']
        labels_loaded = file_loaded[b'labels']
        for it in range (len(data_loaded)):
          self.training_data.append(Image(data_loaded[it], labels_loaded[it]))
    print("Training data loaded successfully!")

  def load_test_data(self):
    print("Loading test data...")
    with open("test/test_batch", 'rb') as file_path:
      file_loaded = pckl.load(file_path, encoding='bytes')
      data_loaded = file_loaded[b'data']
      labels_loaded = file_loaded[b'labels']
      for it in range (len(data_loaded)):
        self.test_data.append(Image(data_loaded[it], labels_loaded[it]))
    print("Test data loaded successfully!")


# Global Methods
def main():
  print("K Nearest Neighbors")
  dm = DataManager()
  dm.load_training_data()
  dm.load_test_data()
  classifier = ImageClassifier(dm.training_data, dm.test_data)
  classifier.manhattan() # Al usar el metodo asi, k por defecto es #1
  """
    classifier.manhattan(2) usa k = 2, y asi con el valor de k deseado.
  """

def tabulate_result(k, hits, success_percentage):
  print(
    tabulate(
      [
        ["k",k],
        ["Hits", hits],
        ["Success", str(success_percentage)+ "%"]
      ],
      headers = ['Variable Name', 'Value'
    ])
  )
