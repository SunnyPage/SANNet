package demo;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.EarlyStopping;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.network.Persistence;
import core.optimization.OptimizationType;
import utils.configurable.DynamicParamException;
import utils.matrix.BinaryFunctionType;
import utils.matrix.ComputableMatrix;
import utils.matrix.DMatrix;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.SMatrix;
import utils.matrix.UnaryFunctionType;
import utils.sampling.BasicSampler;

public class SpeechToSpeech 
{
    public static void main(String [] args) {

        try 
        {
            Run();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void Run()
    {
        NeuralNetwork neuralNetwork;
        try 
        {
            HashMap<Integer, String> dictionaryIndexMapping = new HashMap<>();

            HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> validationInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> validationOutputs = new HashMap<>();
            int maxCountWords = 0;

            HashMap<String, Integer> wordToIndex = new HashMap<>();
            HashMap<Integer, String> indexToword = new HashMap<>();
            ArrayList<String> vocab = new ArrayList<>();           

            //String trainfileName = "C:/Levan/SpeechToSpeech/speech/data_train.csv";
            //String testfileName = "C:/Levan/SpeechToSpeech/speech/data_test.csv"; 
            String trainfileName = "<PATH>/speech/data_train.csv";
            String testfileName = "<PATH>/speech/data_test.csv";           

            maxCountWords = GetDictionary(trainfileName, testfileName, wordToIndex, indexToword, vocab, "\\|", 0);

            HashSet<Integer> inputCols = new HashSet<>();
            HashSet<Integer> outputCols = new HashSet<>();
            for (int i = 1; i <= 352 * 22; i++) inputCols.add(i);
            for (int i = 1; i <= maxCountWords; i++) outputCols.add(0);

            getMNISTData(trainingInputs, trainingOutputs, wordToIndex, indexToword, vocab, trainfileName, dictionaryIndexMapping, maxCountWords, inputCols, outputCols);
            getMNISTData(validationInputs, validationOutputs, wordToIndex, indexToword, vocab, testfileName, dictionaryIndexMapping, maxCountWords, inputCols, outputCols);

            neuralNetwork = buildNeuralNetwork(trainingInputs.get(0).get(0).get(0).getRows(), 22, trainingOutputs.get(0).get(0).get(0).getRows());

            //String persistenceName = "C:/Levan/SpeechToSpeech/speech";
            String persistenceName = "<PATH>/speech";
            
            //neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);                      

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);

            neuralNetwork.setAsClassification();
            neuralNetwork.verboseTraining(10);
            neuralNetwork.setAutoValidate(100);
            neuralNetwork.verboseValidation();
            neuralNetwork.setTrainingEarlyStopping(new TreeMap<>() {{ put(0, new EarlyStopping()); }});

            neuralNetwork.start();

            ///////////////////////////////////////////////////////////////////////
            //Test Predict

            /*TreeMap<Integer, Matrix> currentSample = new TreeMap<>();
            for (int index1 = 0; index1 < 22; index1++) 
            {
                currentSample.put(index1, validationInputs.get(index1).get(0).get(0));
            }

            Matrix nextEncodedWord = neuralNetwork.predictMatrix(currentSample).get(0);
            int TotalRows = validationOutputs.get(0).get(0).get(0).getSubMatrices().get(0).getTotalRows();
            String Textpredicted = "";
            int hh = 0;
            ArrayList<Double> RowsMatrix = new ArrayList<>();
            for (int tt = 0; tt < nextEncodedWord.getTotalRows(); tt++)
            {                  
                if (hh >= TotalRows)   
                {     
                    hh = 0;
                    RowsMatrix.clear();   
                    RowsMatrix = new ArrayList<>();
                } 

                RowsMatrix.add(nextEncodedWord.getValue(tt, 0));

                if (hh == TotalRows - 1)    
                {                    
                    int maximumIndex = -1;
                    Double maximum = 0.0;
                    for (int i = 0; i < RowsMatrix.size(); i++) 
                    {
                        if (maximum < RowsMatrix.get(i))
                        {
                            maximum = RowsMatrix.get(i);
                            maximumIndex = i;
                        }
                    }
    
                    String currentWord = dictionaryIndexMapping.getOrDefault(maximumIndex, "***");                    
                    Textpredicted += currentWord;      
                }                             
                
                hh++;
            } 

            byte[] charset = Textpredicted.getBytes("UTF-8");  
            String newstr = new String(charset, "UTF-8"); 
            System.out.println(newstr);*/
            //////////////////////////////////////////////////////////////////////// 

            neuralNetwork.setTrainingData(new BasicSampler(trainingInputs, trainingOutputs, "randomOrder = false, shuffleSamples = false, sampleSize = 100, numberOfIterations = 300"));
            neuralNetwork.setValidationData(new BasicSampler(validationInputs, validationOutputs, "randomOrder = false, shuffleSamples = false, sampleSize = " + validationInputs.get(0).get(0).get(0).size()));

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            System.out.println("Training...");
            neuralNetwork.printConfusionMatrix(false);
            neuralNetwork.train();

            System.out.println("Finish...");         

            Persistence.saveNeuralNetwork(persistenceName, neuralNetwork);

            neuralNetwork.stop();
        }
        catch (Exception exception) 
        {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    private static void getMNISTData(HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs, HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs, HashMap<String, Integer> wordToIndex, HashMap<Integer, String> indexToword, ArrayList<String> vocab, String fileName, HashMap<Integer, String> dictionaryIndexMapping, int maxCountWords, HashSet<Integer> inputCols,  HashSet<Integer> outputCols) throws MatrixException, FileNotFoundException 
    {       
        readFile(trainingInputs, trainingOutputs, fileName, "\\|", wordToIndex, dictionaryIndexMapping, maxCountWords, inputCols, outputCols, 0, true, true, 352, 22, false, 0, 0);       

        System.out.println(" Done.");
    }

    private static NeuralNetwork buildNeuralNetwork(int inputRowSize, int inputColumnSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException 
    {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        // Encoder and attention layers for processing key, velocity and tick value information.
        int keyAttentionLayerIndex = buildInputAttentionModule(neuralNetworkConfiguration, inputRowSize, inputColumnSize);

        // Final feedforward layers for key, velocity and tick information.
        int keyHiddenLayerIndex = buildOutputAttentionModule(neuralNetworkConfiguration, inputRowSize, inputColumnSize);
        neuralNetworkConfiguration.connectLayers(keyAttentionLayerIndex, keyHiddenLayerIndex);

        // Output layers for key, velocity and tick information.
        int keyOutputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);

        neuralNetworkConfiguration.connectLayers(keyHiddenLayerIndex, keyOutputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        return neuralNetwork;
    }

    public static int GetDictionary(String trainfileName, String testfileName, HashMap<String, Integer> wordToIndex, HashMap<Integer, String> indexToword, ArrayList<String> vocab, String separator, int skipRowsFromStart) throws FileNotFoundException, MatrixException 
    {
        File file_1 = new File(trainfileName);
        Scanner scanner_1 = new Scanner(file_1, "UTF-8");

        File file_2 = new File(testfileName);
        Scanner scanner_2 = new Scanner(file_2, "UTF-8");

        int countSkipRows_2 = 0;
        while (countSkipRows_2++ < skipRowsFromStart && scanner_2.hasNextLine()) scanner_2.nextLine();

        int maxCountWords = 0;

        HashMap<String, Integer> d = new HashMap<>();

        while (scanner_1.hasNextLine()) 
        {
            String Str = "*** " + scanner_1.nextLine().split(separator)[0]; 

            Str = Str.replaceAll("\\.", " . "); 
            Str = Str.replaceAll("\\,", " , ");
            Str = Str.replaceAll("\\?", " ? ");
            Str = Str.replaceAll("\\!", " ! ");
            Str = Str.replaceAll("\\:", " : "); 
            Str = Str.replaceAll("\\;", " ; "); 
            Str = Str.replaceAll("\\'", " ' ");
            Str = Str.replaceAll("\\(", " ( ");
            Str = Str.replaceAll("\\)", " ) ");
            Str = Str.replaceAll("\\[", " [ ");
            Str = Str.replaceAll("\\]", " ] ");
            Str = Str.replaceAll("\\}", " } ");
            Str = Str.replaceAll("\\{", " { ");

            String[] words = Str.split(" ");
            Arrays.setAll(words, index -> words[index].trim());
            Arrays.setAll(words, _index -> words[_index].toLowerCase()); 

            if (maxCountWords < words.length)
            {
                maxCountWords = words.length;
            }
            
            for (int q = 0; q < words.length; q++)
            {
                String word = words[q];
                if (word != "")
                {
                    if (d.containsKey(word) == true)
                    {
                        d.replace(word, d.get(word) + 1);
                    }
                    else
                    {
                        d.put(word, 1);
                    }
                }
            }
        } 

        while (scanner_2.hasNextLine()) 
        {
            String Str = "*** " + scanner_2.nextLine().split(separator)[0]; 

            Str = Str.replaceAll("\\.", " . "); 
            Str = Str.replaceAll("\\,", " , ");
            Str = Str.replaceAll("\\?", " ? ");
            Str = Str.replaceAll("\\!", " ! ");
            Str = Str.replaceAll("\\:", " : "); 
            Str = Str.replaceAll("\\;", " ; "); 
            Str = Str.replaceAll("\\'", " ' ");
            Str = Str.replaceAll("\\(", " ( ");
            Str = Str.replaceAll("\\)", " ) ");
            Str = Str.replaceAll("\\[", " [ ");
            Str = Str.replaceAll("\\]", " ] ");
            Str = Str.replaceAll("\\}", " } ");
            Str = Str.replaceAll("\\{", " { ");

            String[] words = Str.split(" ");
            Arrays.setAll(words, _index -> words[_index].toLowerCase());

            if (maxCountWords < words.length)
            {
                maxCountWords = words.length;
            }
            
            for (int q = 0; q < words.length; q++)
            {
                String word = words[q];
                if (word != "")
                {
                    if (d.containsKey(word) == true)
                    {
                        d.replace(word, d.get(word) + 1);
                    }
                    else
                    {
                        d.put(word, 1);
                    }
                }
            }
        }                             

        int m = 2;
        for(Map.Entry<String, Integer> entry : d.entrySet()) 
        {
            if (entry.getValue() >= 1)
            {
                wordToIndex.put(entry.getKey(), m);
                indexToword.put(m, entry.getKey());
                vocab.add(entry.getKey());
                m++;
            }           
        }  
        
        scanner_1.close();
        scanner_2.close();

        return maxCountWords;
    }

    public static void readFile(HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs, HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs, String fileName, String separator, HashMap<String, Integer> wordToIndex, HashMap<Integer, String> dictionaryIndexMapping, int maxCountWords, HashSet<Integer> inputColumns, HashSet<Integer> outputColumns, int skipRowsFromStart, boolean asSparseMatrix, boolean inAs2D, int inRows, int inCols, boolean outAs2D, int outRows, int outCols) throws FileNotFoundException, MatrixException 
    {
        HashMap<Integer, Integer> inputColumnMap = new HashMap<>();
        HashMap<Integer, Integer> outputColumnMap = new HashMap<>();
        int index;
        index = 0;

        for (Integer pos : inputColumns) inputColumnMap.put(pos, index++);

        index = 0;
        for (Integer pos : outputColumns) outputColumnMap.put(pos, index++);

        inRows = inAs2D ? inRows : inputColumnMap.size();
        inCols = inAs2D ? inCols : 1;

        outRows = outAs2D ? outRows : outputColumnMap.size();

        File file = new File(fileName);
        Scanner scanner = new Scanner(file, "UTF-8");

        int countSkipRows = 0;
        while (countSkipRows++ < skipRowsFromStart && scanner.hasNextLine()) scanner.nextLine();

        HashMap<Matrix, Integer> dictionaryBinaryIndexMapping = new HashMap<>();
        HashMap<String, Matrix> dictionaryStringBinaryIndexMapping = new HashMap<>(); 

        int dictionarySize = wordToIndex.size();
        int maxBits = ComputableMatrix.numberOfBits(wordToIndex.size());                      
        int index_m = 0;
        for(Map.Entry<String, Integer> word : wordToIndex.entrySet()) 
        {
            dictionaryIndexMapping.put(index_m, word.getKey());
            Matrix binaryMatrix = ComputableMatrix.encodeToBitColumnVector(index_m, maxBits);
            dictionaryBinaryIndexMapping.put(binaryMatrix, index_m);
            dictionaryStringBinaryIndexMapping.put(word.getKey(), binaryMatrix);
            index_m++;
        }

        int row = 0;        

        for (int r = 0; r < inCols; r++) 
        {
            trainingInputs.put(r, new HashMap<>());
        }

        for (int r = 0; r < maxCountWords; r++) 
        {
            trainingOutputs.put(r, new HashMap<>());
        }

        trainingOutputs.put(0, new HashMap<>());
        ArrayList<MMatrix> dataSetIn = new ArrayList<>();
        ArrayList<MMatrix> dataSetOut = new ArrayList<>();

        while (scanner.hasNextLine()) 
        {
            String[] items = scanner.nextLine().split(separator);
            
            addItem(trainingInputs, dataSetIn, items, inputColumnMap, row, inAs2D, inRows, inCols, asSparseMatrix);

            addItemText(trainingOutputs, dataSetOut, items, outputColumnMap, row, outAs2D, outRows, asSparseMatrix, dictionaryIndexMapping, dictionaryBinaryIndexMapping, dictionaryStringBinaryIndexMapping, dictionarySize, maxCountWords);

            row++;
        } 

        int trainingIndex = 0;
        for (int i = 0; i < row; i++) 
        {
            for (int j = 0; j < inCols; j++) 
            {
                trainingInputs.get(j).put(trainingIndex, dataSetIn.get(i + j));
                if (j == inCols - 1)
                {
                    trainingIndex++;
                }
            }
        }

        int trainingOutput = 0;
        for (int i = 0; i < row; i++) 
        {
            for (int j = 0; j < maxCountWords; j++) 
            {
                trainingOutputs.get(j).put(trainingOutput, dataSetOut.get(i + j));
                if (j == maxCountWords - 1)
                {
                    trainingOutput++;
                }
            }
        }

        scanner.close();
    }

    private static void addItem(HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs, ArrayList<MMatrix> dataSet,  String[] items, HashMap<Integer, Integer> columnMap, int row, boolean as2D, int rows, int columns, boolean asSparseMatrix) throws MatrixException 
    {     
        String[][] arrayValues = ClockWiseRotation(items, 1, rows, columns);
     
        for (int i = 0; i < arrayValues.length; i++)
        {
            Matrix inItem = !asSparseMatrix ? new DMatrix(rows, 1) : new SMatrix(rows, 1);          

            for (int j = 0; j < arrayValues[0].length; j++)
            {               
                if ((convertToDouble(arrayValues[i][j]) != 0.0) && (convertToDouble(arrayValues[i][j]) != 0))
                {
                    inItem.setValue(j, 0, convertToDouble(arrayValues[i][j]));
                }
                else
                {
                    inItem.setValue(j, 0, 0.1);
                }
            }

           dataSet.add(row, new MMatrix(inItem));           
        }    
    }

    private static void addItemText(HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs,  ArrayList<MMatrix> dataSet, String[] items, HashMap<Integer, Integer> columnMap, int row, boolean as2D, int rows, boolean asSparseMatrix, HashMap<Integer, String> dictionaryIndexMapping, HashMap<Matrix, Integer> dictionaryBinaryIndexMapping, HashMap<String, Matrix> dictionaryStringBinaryIndexMapping, int dictionarySize, int maxCountWords) throws MatrixException 
    {              
        for (Map.Entry<Integer, Integer> entry : columnMap.entrySet()) 
        {
            int pos = entry.getKey();
            
            TextAsBinaryEncoded(dataSet, items[pos], trainingOutputs, row, dictionaryIndexMapping, dictionaryBinaryIndexMapping, dictionaryStringBinaryIndexMapping, dictionarySize, maxCountWords);
        }
    }

    public static void TextAsBinaryEncoded(ArrayList<MMatrix> dataSet, String Str, HashMap<Integer, HashMap<Integer, MMatrix>> data, int row, HashMap<Integer, String> dictionaryIndexMapping, HashMap<Matrix, Integer> dictionaryBinaryIndexMapping, HashMap<String, Matrix> dictionaryStringBinaryIndexMapping, int dictionarySize, int maxCountWords) throws MatrixException 
    {               
        Str = Str.replaceAll("\\.", " . "); 
        Str = Str.replaceAll("\\,", " , ");
        Str = Str.replaceAll("\\?", " ? ");
        Str = Str.replaceAll("\\!", " ! ");
        Str = Str.replaceAll("\\:", " : "); 
        Str = Str.replaceAll("\\;", " ; "); 
        Str = Str.replaceAll("\\'", " ' ");
        Str = Str.replaceAll("\\(", " ( ");
        Str = Str.replaceAll("\\)", " ) ");
        Str = Str.replaceAll("\\[", " [ ");
        Str = Str.replaceAll("\\]", " ] ");
        Str = Str.replaceAll("\\}", " } ");
        Str = Str.replaceAll("\\{", " { ");

        String[] words = Str.split(" ");
        Arrays.setAll(words, index -> words[index].trim());
        Arrays.setAll(words, _index -> words[_index].toLowerCase());                  

        ArrayList<Matrix> encodedWords = new ArrayList<>();
        int count = 0;
        for (String word : words) 
        {
            if (word != "")
            {
                encodedWords.add(dictionaryStringBinaryIndexMapping.get(word));
                count++;
            }
        } 

        for (int i = count; i < maxCountWords; i++)
        {
            encodedWords.add(dictionaryStringBinaryIndexMapping.get("***"));
        }
                
        for (Matrix encodedWord : encodedWords) 
        { 
            dataSet.add(row, new MMatrix(DMatrix.getOneHotVector(dictionarySize, dictionaryBinaryIndexMapping.get(encodedWord))));           
        }   
    }

    private static int buildInputAttentionModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs) throws MatrixException, NeuralNetworkException, DynamicParamException {
        // Encoder layers for input information.
        int[] encoderIndices = new int[numberOfInputs];
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) 
        {
            encoderIndices[inputIndex] = buildInputEncoderModule(neuralNetworkConfiguration, inputIndex, inputSize);
        }

        // Attention layer for input information.
        int combinedIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.INPUT_BASED_ATTENTION);
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            neuralNetworkConfiguration.connectLayers(encoderIndices[inputIndex], combinedIndex);
        }
        return combinedIndex;
    }

    private static int buildOutputAttentionModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs) throws MatrixException, NeuralNetworkException, DynamicParamException {
        // Encoder layers for input information.
        int[] encoderIndices = new int[numberOfInputs];
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) 
        {
            encoderIndices[inputIndex] = buildOutputEncoderModule(neuralNetworkConfiguration, inputIndex, inputSize);
        }

        // Attention layer for input information.
        int combinedIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.INPUT_BASED_ATTENTION);
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) 
        {
            neuralNetworkConfiguration.connectLayers(encoderIndices[inputIndex], combinedIndex);
        }
        return combinedIndex;
    }

    /**
     * Builds bi-directional RNN module.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param inputIndex input index
     * @param inputWidth input width
     * @return index of module output layer.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static int buildInputEncoderModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputIndex, int inputWidth) throws NeuralNetworkException, DynamicParamException, MatrixException 
    {
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputWidth);
        int positionalEmbeddingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.POSITIONAL_ENCODING, "positionIndex = " + inputIndex);
        int feedforwardLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.TANH), "width = " + inputWidth);
        int passLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, positionalEmbeddingLayerIndex);
        neuralNetworkConfiguration.connectLayers(positionalEmbeddingLayerIndex, feedforwardLayerIndex);
        neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex, passLayerIndex);
        neuralNetworkConfiguration.connectLayers(positionalEmbeddingLayerIndex, passLayerIndex);
        return passLayerIndex;
    }

    private static int buildOutputEncoderModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputIndex, int inputWidth) throws NeuralNetworkException, DynamicParamException, MatrixException {
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputWidth);
        int feedforwardLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.TANH), "width = " + inputWidth);
        int positionalEmbeddingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.POSITIONAL_ENCODING, "positionIndex = " + inputIndex);
        int passLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT);
        
        neuralNetworkConfiguration.connectLayers(positionalEmbeddingLayerIndex, passLayerIndex);
        neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex, passLayerIndex);
        neuralNetworkConfiguration.connectLayers(positionalEmbeddingLayerIndex, feedforwardLayerIndex);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, positionalEmbeddingLayerIndex);         
        
        return passLayerIndex;
    }

    public static String[][] ClockWiseRotation(String[] data, int StartPos, int column, int row) 
    {
        int ColumnLeng = (data.length - StartPos) / row;
        String a[][] = new String[ColumnLeng][row];
        int dd = StartPos;
        for (int v = 0; v < ColumnLeng; v++)
        {
            for (int j = 0; j < row; j++) 
            {
                a[v][j] = data[dd];
                dd++;                             
            }
        }
        
        return resize(rotateMatrixBy90DegreeClockwise(a), column, row);     
    }

    static String[][] resize(String[][] matrix, int w, int h) 
    {
        String[][] temp = new String[h][w];
        for (int i = 0; i < temp.length; i++)
        {
            for (int j = 0; j < temp[0].length; j++)
            {
                temp[i][j] = "0.1";
            }
        }

        h = Math.min(h, matrix.length);
        w = Math.min(w, matrix[0].length);
        for (int i = 0; i < h; i++)
            System.arraycopy(matrix[i], 0, temp[i], 0, w);
        return temp;
    }
    private static String[][] rotateMatrixBy90DegreeClockwise(String[][] matrix) 
    {
        int totalRowsOfRotatedMatrix = matrix[0].length; //Total columns of Original Matrix
        int totalColsOfRotatedMatrix = matrix.length; //Total rows of Original Matrix
       
        String[][] rotatedMatrix = new String[totalRowsOfRotatedMatrix][totalColsOfRotatedMatrix];
       
        for (int i = 0; i < matrix.length; i++) 
        {
            for (int j = 0; j < matrix[0].length; j++) 
            {
                rotatedMatrix[j][ (totalColsOfRotatedMatrix-1)- i] = matrix[i][j]; 
            }
        }
        return rotatedMatrix;
    }
       
    //Rotate Matrix to 90 degree toward Left(counter clockwise)
    private static String[][] rotateMatrixBy90DegreeCounterClockwise(String[][] matrix) 
    {      
        int totalRowsOfRotatedMatrix = matrix[0].length; //Total columns of Original Matrix
        int totalColsOfRotatedMatrix = matrix.length; //Total rows of Original Matrix
        
        String[][] rotatedMatrix = new String[totalRowsOfRotatedMatrix][totalColsOfRotatedMatrix];
        
        for (int i = 0; i < matrix.length; i++) 
        {
            for (int j = 0; j < matrix[0].length; j++) 
            {
                rotatedMatrix[(totalRowsOfRotatedMatrix-1)-j][i] = matrix[i][j]; 
            }
        }
        return rotatedMatrix;
    }

    /**
     * Converts string to double value.
     *
     * @param item string to be converted.
     * @return converted double value.
     */
    private static Double convertToDouble(String item) {
        double value = 0;
        try {
            value = Double.parseDouble(item);
        }
        catch (NumberFormatException numberFormatException) {
        }
        return value;
    }

}
