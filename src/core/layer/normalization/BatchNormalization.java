/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.normalization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.node.Node;
import utils.sampling.Sequence;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Implements layer for batch normalization.
 *
 */
public class BatchNormalization extends AbstractExecutionLayer {

    /**
     * Parameter name types for batch normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *     - momentum: degree of weighting decrease for exponential moving average. (default value 0.95).<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN), " +
            "(momentum:DOUBLE)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class BatchNormalizationWeightSet implements WeightSet, Serializable {

        /**
         * Learnable parameter gamma of batch normalization layer.
         *
         */
        private final Matrix gamma;

        /**
         * Learnable parameter beta of batch normalization layer.
         *
         */
        private final Matrix beta;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *  @param previousLayerWidth width of previous layer.
         * @param previousLayerHeight height of previous layer.
         */
        BatchNormalizationWeightSet(int previousLayerWidth, int previousLayerHeight) {
            gamma = new DMatrix(previousLayerWidth, previousLayerHeight, (row, col) -> new Random().nextGaussian() * 0.1);
            gamma.setName("Gamma");
            beta = new DMatrix(previousLayerWidth, previousLayerHeight);
            beta.setName("Beta");

            weights.add(gamma);
            weights.add(beta);

            registerWeight(gamma, false, false);
            registerWeight(beta, false, false);
        }

        /**
         * Returns set of weights.
         *
         * @return set of weights.
         */
        public HashSet<Matrix> getWeights() {
            return weights;
        }

        /**
         * Reinitializes weights.
         *
         */
        public void reinitialize() {
            gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
            beta.reset();
        }

        /**
         * Returns number of parameters.
         *
         * @return number of parameters.
         */
        public int getNumberOfParameters() {
            int numberOfParameters = 0;
            for (Matrix weight : weights) numberOfParameters += weight.size();
            return numberOfParameters;
        }

    }

    /**
     * Weight set.
     *
     */
    protected BatchNormalizationWeightSet weightSet;

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Matrix for epsilon value.
     *
     */
    private Matrix epsilonMatrix;

    /**
     * Stores rolling average mean of samples when neural network is in training mode.<br>
     * Stored rolling average mean is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix averageMean;

    /**
     * Mean reference matrix.
     *
     */
    private transient Matrix mean;

    /**
     * Node for mean.
     *
     */
    private Node meanNode;

    /**
     * Stores rolling average variance of samples when neural network is in training mode.<br>
     * Stored rolling average variance is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix averageVariance;

    /**
     * Variance reference matrix.
     *
     */
    private transient Matrix variance;

    /**
     * Node for variance.
     *
     */
    private Node varianceNode;

    /**
     * Sample size for a batch. Double is used for calculation purposes.
     *
     */
    private double batchSize;

    /**
     * Momentum value for exponential moving average.
     *
     */
    private double momentum;

    /**
     * Constructor for batch normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for batch normalization layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        epsilonMatrix = new DMatrix(10E-8);
        epsilonMatrix.setName("Epsilon");
        meanOnly = false;
        batchSize = -1;
        momentum = 0.95;
    }

    /**
     * Returns parameters used for batch normalization layer.
     *
     * @return parameters used for batch normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + BatchNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for batch normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *     - momentum: degree of weighting decrease for exponential moving average. (default value 0.95).<br>
     *
     * @param params parameters used for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
        if (params.hasParam("momentum")) momentum = params.getValueAsDouble("momentum");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return false;
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return weightSet;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        weightSet = new BatchNormalizationWeightSet(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight());
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new MMatrix(1, "Inputs");
        Matrix input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, input);
        return new TreeMap<>() {{ put(0, inputs); }};
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        super.defineProcedure();

        meanNode = procedure.getNode(mean);
        varianceNode = procedure.getNode(variance);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException, DynamicParamException {
        mean = inputs.mean();

        MMatrix outputs;
        MMatrix meanNormalizedInput = inputs.subtract(mean);
        if (!meanOnly) {
            variance = inputs.variance(mean);
            outputs = meanNormalizedInput.divide(variance.add(epsilonMatrix).apply(UnaryFunctionType.SQRT)).multiply(weightSet.gamma).add(weightSet.beta);
        }
        else outputs = meanNormalizedInput.multiply(weightSet.gamma).add(weightSet.beta);
        outputs.setName("Output", true);

        return outputs;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>() {{ add(epsilonMatrix); }};
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>() {{ add(epsilonMatrix); }};
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        Sequence inputSequence = getDefaultLayerInput();

        if (isTraining()) {
            if (batchSize == -1) batchSize = inputSequence.sampleSize();
            if (inputSequence.sampleSize() < 2) throw new MatrixException("Batch normalization must have minimum batch size of 2 for training phase.");

            super.forwardProcess();

            averageMean = meanNode.getMatrix().exponentialMovingAverage(averageMean, momentum);
            if (!meanOnly) averageVariance = varianceNode.getMatrix().exponentialMovingAverage(averageVariance, momentum);
        }
        else {
            Sequence layerOutputs = new Sequence();
            if (!meanOnly) {
                Matrix averageStandardDeviation = averageVariance.multiply(batchSize / (batchSize - 1)).add(epsilonMatrix).apply(UnaryFunctionType.SQRT);
                for (Map.Entry<Integer, MMatrix> entry : inputSequence.entrySet()) {
                    int sampleIndex = entry.getKey();
                    MMatrix inputSample = entry.getValue();
                    int depth = inputSample.getDepth();
                    MMatrix outputSample = new MMatrix(depth);
                    layerOutputs.put(sampleIndex, outputSample);
                    for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                        outputSample.put(depthIndex, inputSample.get(depthIndex).subtract(averageMean).divide(averageStandardDeviation).multiply(weightSet.gamma).add(weightSet.beta));
                    }
                }
            }
            else {
                for (Map.Entry<Integer, MMatrix> entry : inputSequence.entrySet()) {
                    int sampleIndex = entry.getKey();
                    MMatrix inputSample = entry.getValue();
                    MMatrix outputSample = new MMatrix(inputSequence.get(sampleIndex).getDepth());
                    layerOutputs.put(sampleIndex, outputSample);
                    int depth = inputSample.getDepth();
                    for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                        outputSample.put(depthIndex, inputSample.get(depthIndex).subtract(averageMean).multiply(weightSet.gamma).add(weightSet.beta));
                    }
                }
            }
            setLayerOutputs(layerOutputs);
        }
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Mean only: " + meanOnly + ", Momentum: " + momentum;
    }

}
