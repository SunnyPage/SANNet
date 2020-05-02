/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package demo;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.Metrics;
import core.metrics.MetricsType;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.preprocess.ReadCSVFile;
import utils.DynamicParamException;
import utils.Persistence;
import utils.Sample;
import utils.Sequence;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

/**
 * Class that implements learning and prediction of MNIST digits by using Convolutional Neural Network (CNN).<br>
 * Implementation reads training and test samples from CSV file and then executes learning process.<br>
 *
 */
public class MNISTDemo {

    /**
     * Main function for demo.<br>
     * Function reads samples from files, build CNN and executed training, validation and testing.<br>
     *
     * @param args input arguments (not used).
     */
    public static void main(String [] args) {

        NeuralNetwork neuralNetwork;
        try {
            HashMap<Integer, LinkedHashMap<Integer, Sample>> trainMNIST = getMNISTData(true);
            HashMap<Integer, LinkedHashMap<Integer, Sample>> testMNIST = getMNISTData(false);

            neuralNetwork = buildNeuralNetwork(trainMNIST.get(0).get(0).get(0).getRows(), trainMNIST.get(1).get(0).get(0).getRows());

            String persistenceName = "/home/jack/Downloads/MNIST_NN";
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);

            neuralNetwork.setTaskType(MetricsType.CLASSIFICATION, false);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.setAutoValidate(100);
            neuralNetwork.verboseValidation();

            neuralNetwork.start();

            neuralNetwork.setTrainingData(new BasicSampler(trainMNIST.get(0), trainMNIST.get(1), "randomOrder = true, shuffleSamples = true, sampleSize = 50"));
            neuralNetwork.setValidationData(new BasicSampler(testMNIST.get(0), testMNIST.get(1), "randomOrder = true, shuffleSamples = true, sampleSize = 50"));
            neuralNetwork.setTrainingIterations(10000);

            System.out.println("Training...");
            neuralNetwork.train();

            System.out.println("Predicting...");
            Metrics predictionMetrics = new Metrics(MetricsType.CLASSIFICATION);
            Sequence predict = neuralNetwork.predict(new Sequence(testMNIST.get(0)));
            predictionMetrics.report(predict, new Sequence(testMNIST.get(1)));
            predictionMetrics.store(1, false);
            predictionMetrics.printReport();
            predictionMetrics.printConfusionMatrix();

            Persistence.saveNeuralNetwork(persistenceName, neuralNetwork);

            neuralNetwork.stop();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Function that builds Convolutional Neural Network (CNN).<br>
     * MNIST images (size 28x28) are used as inputs.<br>
     * Outputs are 10 discrete outputs where each output represents single digit.<br>
     *
     * @param inputSize input size of convolutional neural network.
     * @param outputSize output size of convolutional neural network.
     * @return CNN instance.
     * @throws DynamicParamException throws exception is setting of parameters fails.
     * @throws NeuralNetworkException throws exception if creation of CNN fails.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws MatrixException, DynamicParamException, NeuralNetworkException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = 28, height = 28");
        neuralNetwork.addHiddenLayer(LayerType.CONVOLUTIONAL, new ActivationFunction(UnaryFunctionType.RELU, "alpha = 0.01"), Init.UNIFORM_XAVIER_CONV, "filters = 16, filterSize = 3, stride = 1, asConvolution = false");
//        neuralNetwork.addHiddenLayer(LayerType.POOLING, "poolSize = 2, stride = 1, avgPool = false");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU, "alpha = 0.01"), "width = 40");
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.AMSGRAD);
        neuralNetwork.addNormalizer(2, NormalizationType.BATCH_NORMALIZATION);
        neuralNetwork.setLossFunction(BinaryFunctionType.CROSS_ENTROPY);
        return neuralNetwork;
    }

    /**
     * Reads MNIST samples from CSV files.<br>
     * MNIST training set consist of 60000 samples and test set of 10000 samples.<br>
     * First column is assumed to be outputted digits (value 0 - 9).<br>
     * Next 784 (28x28) columns are assumed to be input digit (gray scale values 0 - 255).<br>
     *
     * @param trainSet if true training set file is read otherwise test set file is read.
     * @return encoded input and output pairs.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws FileNotFoundException throws exception if matrix cannot be read.
     */
    private static HashMap<Integer, LinkedHashMap<Integer, Sample>> getMNISTData(boolean trainSet) throws MatrixException, FileNotFoundException {
        System.out.print("Loading " + (trainSet ? "training" : "test") + " data... ");
        HashSet<Integer> inputCols = new HashSet<>();
        HashSet<Integer> outputCols = new HashSet<>();
        for (int i = 1; i < 785; i++) inputCols.add(i);
        outputCols.add(0);
        String fileName = trainSet ? "/home/jack/Downloads/mnist_train.csv" : "/home/jack/Downloads/mnist_test_mini.csv";
        HashMap<Integer, LinkedHashMap<Integer, Sample>> data = ReadCSVFile.readFile(fileName, ",", inputCols, outputCols, 0, true, true, 28, 28, false, 0, 0);
        for (Sample sample : data.get(0).values()) {
            for (Matrix entry : sample.values()) {
                entry.divide(255, entry);
            }
        }
        for (Sample sample : data.get(1).values()) {
            for (Integer entryIndex : sample.keySet()) {
                int value = (int)sample.get(entryIndex).getValue(0,0);
                Matrix output = new SMatrix(10, 1);
                output.setValue(value, 0, 1);
                sample.put(entryIndex, output);
            }
        }
        System.out.println(" Done.");
        return data;
    }

}
