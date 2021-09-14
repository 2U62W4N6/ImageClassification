import numpy as np
import matplotlib.pyplot as plt
import helper
import os

np.random.seed(1)


class TwoLayerNeuralNetwork:
    """
    Ein 2-Layer 'fully-connected' neural network, d.h. alle Neuronen sind mit allen anderen
    verbunden. Die Anzahl der Eingabevektoren ist N mit einer Dimension D, einem 'Hidden'-Layer mit
    H Neuronen. Es soll eine Klassifikation über 10 Klassen (C) durchgeführt werden.
    Wir trainieren das Netzwerk mit einer Kreuzentropie-Loss Funktion. Das Netzwerk nutzt ReLU 
    Aktivierungsfunktionen nach dem ersten Layer.
    Die Architektur des Netzwerkes läßt sich abstrakt so darstellen:
    Eingabe - 'fully connected'-Layer - ReLU - 'fully connected'-Layer - Softmax

    Die Ausgabe aus dem 2.Layer sind die 'Scores' (Wahrscheinlichkeiten) für jede Klasse.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Intitialisierung des Netzes - Die Gewichtungsmatrizen und die Bias-Vektoren werden mit
        Zufallswerten initialisiert.
        W1: 1.Layer Gewichte (D, H)
        b1: 1.Layer Bias (H,)
        W2: 2.Layer Gewichte (H, C)
        b2: 2.Layer Bias (C,)

        :param input_size: Die CIFAR-10 Bilder haben die Dimension D (32*32*3).
        :param hidden_size: Anzahl der Neuronen im Hidden-Layer H.
        :param output_size: Anzahl der Klassen C.
        :param std: Skalierungsfaktoren für die Initialisierung (muss klein sein)
        :return:
        """
        self.W1 = std * np.random.randn(input_size, hidden_size)
        self.b1 = std * np.random.randn(1, hidden_size)
        self.W2 = std * np.random.randn(hidden_size, output_size)
        self.b2 = std * np.random.randn(1, output_size)

    def softmax(self, z):
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_derivative(self, output):
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def loss_deriv_softmax(self, activation, y_batch):
        batch_size = y_batch.shape[0]
        dCda2 = activation
        dCda2[range(batch_size), y_batch] -= 1
        dCda2 /= batch_size
        return dCda2

    def loss_crossentropy(self, activation, y_batch):
        """
        Berechnet den loss (Fehler) des 2-Layer-Netzes

        :param activation: Aktivierungen / Output des Netzes
        :param y_batch: Vektor mit den Trainingslabels y[i] für einen Batch enthält ein Label aus X[i] 
                        und jedes y[i] ist ein Integer zwischen 0 <= y[i] < C (Anzahl der Klassen),
        :return: loss - normalisierter Fehler des Batches
        """

        batch_size = y_batch.shape[0]
        correct_logprobs = -np.log(activation[range(batch_size), y_batch])
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def forward(self, X, y):
        """
        Führt den gesamten Forward Prozess durch und berechnet den Fehler (loss) und die Aktivierungen a1 und
        a2 und gibt diese Werte zuruück
        :param X: Trainings bzw. Testset
        :param y: Labels des Trainings- bzw. Testsets
        :return: loss, a1, a2
        """

        # Berechen Sie den score
        N, D = X.shape
        
        m1 = np.dot(X, self.W1)
        m1b1 = m1 + self.b1
        a1 = self.relu(m1b1)

        m2 = np.dot(a1, self.W2)
        m2b2 = m2 + self.b2
        a2 = self.softmax(m2b2)

        loss = self.loss_crossentropy(a2, y)
        # TODO: Berechnen Sie den Forward-Schritt und geben Sie den Vektor mit Scores zurueck
        # Nutzen Sie die ReLU Aktivierungsfunktion im ersten Layer
        # Berechnen Sie die Klassenwahrscheinlichkeiten unter Nutzung der softmax Funktion

        # TODO: Berechnen Sie den Fehler mit der cross-entropy Funktion
        return loss, a1, a2
        # return loss, a1, a2

    def backward(self, a1, a2, X, y):
        """
        Backward pass- dabei wird der Gradient der Gewichte W1, W2 und der Biases b1, b2 aus den Ausgaben des Netzes
        berechnet und die Gradienten der einzelnen Layer als ein Dictionary zurückgegeben.
        Zum Beispiel sollte grads['W1'] die Gradienten von self.W1 enthalten (das ist eine Matrix der gleichen Größe
        wie self.W1.
        :param a1: Aktivierung aus dem 1.Layer
        :param a2: Aktivierung aus dem 2.Layer -> Output des Netzes
        :param X:
        :param y:
        :return:
        """


        # Backward pass: Berechnen Sie die Gradienten
        N, D = X.shape

        # Füllen Sie das Dictionary grads['W2'], grads['b2'], grads['W1'], grads['b1']
        grads = {}

        dCda2 = self.loss_deriv_softmax(a2,y) 
        da2da1 = self.W2 
        da1dm1 = self.relu_derivative(a1) 
        dm1dW1 = X 
        da2dW2 = a1 

        tmp1 = np.dot(dCda2, da2da1.T)
        tmp2 = tmp1 * da1dm1 
        dCdW1 = np.dot(dm1dW1.T, tmp2)
        dCdW2 = np.dot(da2dW2.T, dCda2)

        dcdb2 = np.sum(dCda2, axis=0, keepdims=True)
        dcdb1 = np.sum(tmp2, axis=0, keepdims=True)

        # Nutzen Sie dabei die Notizen aus der Vorlesung und die gegebenen Ableitungsfunktionen

        grads['W1'] = learning_rate * dCdW1
        grads['b1'] = learning_rate * dcdb1

        grads['W2'] = learning_rate * dCdW2
        grads['b2'] = learning_rate * dcdb2
        return grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95, num_iters=100,
              batch_size=200, verbose=False):
        """
        Training des Neuronalen Netzwerkes unter der Nutzung des iterativen
        Optimierungsverfahrens Stochastic Gradient Descent
        Train this neural network using stochastic gradient descent.

        :param X: Numpy Array der Größe (N,D)
        :param y: Numpy Array der Größe (N,) mit den jeweiligen Labels y[i] = c. Das bedeutet, dass X[i] das label c hat
                  mit 0 <= c < C
        :param X_val: Numpy Array der Größe (N_val,D) mit den Validierungs-/Testdaten
        :param y_val: Numpy Array der Größe (N_val,) mit den Labels für die Validierungs-/Testdaten
        :param learning_rate: Faktor der Lernrate für den Optimierungsprozess
        :param learning_rate_decay: gibt an, inwieweit die Lernrate in jeder Epoche angepasst werden soll
        :param num_iters: Anzahl der Iterationen der Optimierung
        :param batch_size: Anzahl der Trainingseingabebilder, die in jedem forward-Schritt mitgegeben werden sollen
        :param verbose: boolean, ob etwas ausgegeben werden soll
        :return: dict (fuer die Auswertung) - enthält Fehler und Genauigkeit der Klassifizierung für jede Iteration bzw. Epoche
        """
        num_train = X.shape[0]
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        # Wir nutzen einen Stochastischen Gradient Decent (SGD) Optimierer um unsere
        # Parameter W1,W2,b1,b2 zu optimieren
        loss_history = []
        loss_val_history = []
        train_acc_history = []
        val_acc_history = []

        sample_propabilities = np.ones(X.shape[0])
        for it in range(num_iters):

            batch_indices = np.random.choice(num_train, size=batch_size)

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            ############################
            # TODO: Erzeugen Sie einen zufälligen Batch der Größe batch_size
            # aus den Trainingsdaten und speichern diese in X_batch und y_batch
            # X_batch
            # y_batch


            ############################

            # TODO: Berechnung von loss und gradient für den aktuellen Batch
            loss, a1 , a2 = self.forward(X_batch, y_batch)
            # Merken des Fehlers für den Plot
            loss_history.append(loss)
            # Berechnung des Fehlers mit den aktuellen Parametern (W, b)
            # mit dem Testset
            loss_val, a1_val, a2_val = self.forward(X_val, y_val)
            loss_val_history.append(loss_val)

            ############################
            # TODO: Nutzen Sie die Gradienten aus der Backward-Funktion und passen
            # Sie die Parameter an (self.W1, self.b1 etc). Diese werden mit der Lernrate
            # gewichtet
            grads = self.backward(a1, a2, X_batch, y_batch)

            self.W1 -= grads['W1']
            self.W2 -= grads['W2']
            self.b1 -= grads['b1']
            self.b2 -= grads['b2']

            ############################

            # Ausgabe des aktuellen Fehlers. Diese sollte am Anfang erstmal nach unten gehen
            # kann aber immer etwas schwanken.
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Wir überprüfen jede Epoche die Genauigkeit (von Trainingsset und Testset)
            # und dämpfen die Lernrate
            if it % iterations_per_epoch == 0:
                # Überprüfung der Klassifikationsgenauigkeit
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print('epoch done... acc', val_acc)

                # Dämpfung der Lernrate
                learning_rate *= learning_rate_decay

        # Zum Plotten der Genauigkeiten geben wir die
        # gesammelten Daten zurück
        return {
            'loss_history': loss_history,
            'loss_val_history': loss_val_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Benutzen Sie die trainierten Gewichte des 2-Layer-Netzes um die Klassen für das
        Validierungsset vorherzusagen. Dafür müssen Sie für das/die Eingabebilder X nur
        die Scores berechnen. Der höchste Score ist die vorhergesagte Klasse. Dieser wird in y_pred
        geschrieben und zurückgegeben.

        :param X: Numpy Array der Größe (N,D)
        :return: y_pred Numpy Array der Größe (N,) die die jeweiligen Labels für alle Elemente in X enthaelt.
        y_pred[i] = c bedeutet, das fuer X[i] die Klasse c mit 0<=c<C vorhergesagt wurde
        """
        y_pred = None

        ############################
        # TODO: Implementieren Sie die Vorhersage. D.h. für ein/mehrere Bild/er mit den gelernten
        # Parametern den Wahrscheinlichkeit berechnen.
        # np.argmax in dem Wahrscheinlichkeitsvektor ist die wahrscheinlichste Klasse
        ############################
        # Implementieren Sie nochmals den Forward pass um die Wahrscheinlichkeiten
        # vorherzusagen
        #y_pred = np.argmax(a2, axis=1)
        m1 = X.dot(self.W1)
        mb1 = m1+self.b1
        a1 = self.relu(mb1)
        m2 = a1.dot(self.W2)
        mb2 = m2+self.b2
        a2 = self.softmax(mb2)

        y_pred = np.argmax(a2, axis=1)

        return y_pred


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = helper.prepare_CIFAR10_images()
    # TODO: Laden der Bilder. Hinweis - wir nutzen nur die Trainigsbilder zum Trainieren und die
    # Validierungsbilder zum Testen.
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    # Grösse der Bilder
    input_size = 32 * 32 * 3
    # Anzahl der Klassen
    num_classes = 10

    #############################################
    # Hyperparameter
    #############################################

    # TODO: mit diesen Parametern sollten Sie in etwa auf eine
    # Klassifikationsgenauigkeit von 43% kommen. Optimieren Sie die
    # Hyperparameter um die Genauigkeit zu erhöhen (bitte tun sie das
    # systematisch und nicht einfach durch probieren - also z.B. in einem
    # for-loop eine Reihe von Parametern testen und die Einzelbilder abspeichern)

    # beste parameter ausgewählt
    hidden_sizes = [400] # [100, 200, 300, 400, 500]  Anzahl der Neuronen im Hidden Layer
    num_iters = [3000]  # [1000, 2000, 3000, 4000] Anzahl der Optimierungsiterationen
    batch_sizes = [300]  # [100, 200, 300, 400] Eingabeanzahl der Bilder
    learning_rates = [0.005]  #[0.001, 0.005, 0.01] Lernrate
    learning_rate_decays = [0.75]  # [0.75, 0.85, 0.95] Lernratenabschwächung

    best_acc = 0
    best_settings = {}
    best_net = None
    best_stats = None

    for learning_rate_decay in learning_rate_decays:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for hidden_size in hidden_sizes:
                    for num_iter in num_iters:
                        net = TwoLayerNeuralNetwork(input_size, hidden_size, num_classes)

                        # Train the network
                        stats = net.train(X_train, y_train, X_val, y_val,
                                        num_iters=num_iter, batch_size=batch_size,
                                        learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, verbose=True)

                        if best_acc <= stats['val_acc_history'][-1]:
                            best_acc = stats['val_acc_history'][-1]
                            best_settings['learning_rate_decay'] = learning_rate_decay
                            best_settings['learning_rate'] = learning_rate
                            best_settings['batch_size'] = batch_size
                            best_settings['hidden_size'] = hidden_size
                            best_settings['num_iter'] = num_iter
                            best_net = net
                            best_stats = stats

                        with open('result.txt', 'a') as myfile:
                            result = f"hidden_size: {hidden_size}\n"
                            result += f"num_iter: {num_iter}\n"
                            result += f"batch_size: {batch_size}\n"
                            result += f"learning_rate: {learning_rate}\n"
                            result += f"learning_rate_decay: {learning_rate_decay}\n"
                            result += "result: \n"
                            result += f"Final training loss:  {stats['loss_history'][-1]}\n"
                            result += f"Final validation loss: {stats['loss_val_history'][-1]}\n"
                            result += f"Final validation accuracy: {stats['val_acc_history'][-1]}\n"
                            result += "__________________________\n"
                            myfile.write(result)


    print('best_acc: ', best_acc)
    print('best settings: ', best_settings)
    helper.plot_net_weights(best_net)
    helper.plot_accuracy(best_stats)
    helper.plot_loss(best_stats)
