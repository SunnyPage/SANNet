package core.layer.reinforcement;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;

/**
 * Implements dueling layer for Deep Q Network.
 *
 */
public class DuelingLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for dueling layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class DuelingWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 7859757124635021579L;

        /**
         * Weight matrix for value.
         *
         */
        private final Matrix valueWeight;

        /**
         * Bias matrix for value.
         *
         */
        private final Matrix valueBias;

        /**
         * Weight matrix for action.
         *
         */
        private final Matrix actionWeight;

        /**
         * Bias matrix for bias.
         *
         */
        private final Matrix actionBias;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param previousLayerWidth width of previous layer.
         * @param layerWidth width of current layer.
         * @param regulateDirectWeights if true direct weights are regulated.
         */
        DuelingWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights) {
            valueWeight = new DMatrix(1, previousLayerWidth, initialization, "ValueWeight");
            valueBias = new DMatrix(layerWidth, 1, "ValueBias");
            actionWeight = new DMatrix(layerWidth, previousLayerWidth, initialization, "ActionWeight");
            actionBias = new DMatrix(layerWidth, 1, "ActionBias");

            weights.add(valueWeight);
            weights.add(valueBias);
            weights.add(actionWeight);
            weights.add(actionBias);

            registerWeight(valueWeight, regulateDirectWeights, true);
            registerWeight(valueBias, false, false);
            registerWeight(actionWeight, regulateDirectWeights, true);
            registerWeight(actionBias, false, false);
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
            valueWeight.initialize(initialization);
            valueBias.reset();
            actionWeight.initialize(initialization);
            actionBias.reset();
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
    protected DuelingWeightSet weightSet;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Constructor for dueling layer.
     *
     * @param layerIndex layer index
     * @param params parameters for dueling layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DuelingLayer(int layerIndex, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, null, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        regulateDirectWeights = true;
    }

    /**
     * Returns parameters used for dueling layer.
     *
     * @return parameters used for dueling layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + DuelingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for dueling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     * @param params parameters used for dueling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
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
        return true;
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
        weightSet = new DuelingWeightSet(initialization, getPreviousLayerWidth(), super.getLayerWidth(), regulateDirectWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new MMatrix(2, "Inputs");

        Matrix valueInput = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (getPreviousLayer().isBidirectional()) valueInput = valueInput.split(getPreviousLayerWidth() / 2, true);
        inputs.put(0, valueInput);

        Matrix actionInput = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (getPreviousLayer().isBidirectional()) actionInput = actionInput.split(getPreviousLayerWidth() / 2, true);
        inputs.put(1, actionInput);

        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        Matrix valueOutput = weightSet.valueWeight.dot(inputs.get(0));
        valueOutput = valueOutput.add(weightSet.valueBias);

        Matrix actionOutput = weightSet.actionWeight.dot(inputs.get(1));
        actionOutput = actionOutput.add(weightSet.actionBias);

        Matrix output = valueOutput.add(actionOutput.subtract(actionOutput.meanAsMatrix()));

        output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);
        return outputs;

    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        resetLayer();
        int inputDepth = 2;
        Sequence previousLayerOutputs = getPreviousLayerOutputs();
        Sequence layerInputs = new Sequence();
        for (Integer sampleIndex : previousLayerOutputs.keySet()) {
            MMatrix previousLayerOutput = previousLayerOutputs.get(sampleIndex);
            MMatrix layerInput = new MMatrix(inputDepth, "Inputs");
            layerInput.put(0, previousLayerOutput.get(0));
            layerInput.put(1, previousLayerOutput.get(0));
            layerInputs.put(sampleIndex, layerInput);
        }
        if (procedure != null) setLayerOutputs(procedure.calculateExpression(layerInputs));
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        if (procedure != null) setLayerGradients(procedure.calculateGradient(getNextLayerGradients(), getTruncateSteps()));
        int inputDepth = getLayerGradients().getDepth();
        Sequence layerGradients = getLayerGradients();
        for (Integer sampleIndex : layerGradients.keySet()) {
            MMatrix layerGradient = layerGradients.get(sampleIndex);
            MMatrix currentLayerGradient = new MMatrix(inputDepth);
            layerGradient.get(0).add(layerGradient.get(1), layerGradient.get(0));
            currentLayerGradient.put(0, layerGradient.get(0));
            layerGradients.put(sampleIndex, currentLayerGradient);
        }
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
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
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "N/A";
    }

}