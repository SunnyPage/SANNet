/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;
import utils.sampling.Sequence;

import java.util.HashMap;

/**
 * Implements bidirectional basic simple recurrent layer.<br>
 * This layer is by nature prone to numerical instability with limited temporal memory.<br>
 *
 */
public class BiRecurrentLayer extends RecurrentLayer {

    /**
     * Reverse weight set.
     *
     */
    private RecurrentWeightSet reverseWeightSet = null;

    /**
     * Reverse procedure for layer. Procedure contains chain of forward and backward expressions.
     *
     */
    protected Procedure reverseProcedure = null;

    /**
     * Layer outputs for reverse sequence.
     *
     */
    private transient Sequence reverseLayerOutputs;

    /**
     * Layer gradients for reverse sequence.
     *
     */
    private transient Sequence reverseLayerGradients;

    /**
     * Constructor for Bidirectional recurrent layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiRecurrentLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, activationFunction, initialization, params);
    }

    /**
     * Check if layer is bidirectional.
     *
     * @return true if layer is bidirectional otherwise returns false.
     */
    public boolean isBidirectional() {
        return true;
    }

    /**
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getLayerWidth() {
        return 2 * super.getLayerWidth();
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        super.initializeWeights();
        reverseWeightSet = new RecurrentWeightSet(initialization, getPreviousLayerWidth(), super.getLayerWidth(), getRegulateDirectWeights(), getRegulateRecurrentWeights());
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        super.defineProcedure();
        currentWeightSet = reverseWeightSet;
        reverseProcedure = new ProcedureFactory().getProcedure(this, reverseWeightSet.getWeights(), getConstantMatrices(), getStopGradients(), true);
    }

    /**
     * Resets outputs of neural network layer.
     *
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void resetLayerOutputs() throws MatrixException {
        super.resetLayerOutputs();
        reverseLayerOutputs = new Sequence(getLayerDepth());
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        resetLayer();
        procedure.calculateExpression(getPreviousLayerOutputs(), getLayerOutputs());
        reverseProcedure.calculateExpression(getPreviousLayerOutputs(), reverseLayerOutputs);
        setLayerOutputs(getLayerOutputs().join(reverseLayerOutputs, true));
    }

    /**
     * Resets gradients of neural network layer.
     *
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void resetLayerGradients() throws MatrixException {
        super.resetLayerGradients();
        reverseLayerGradients = new Sequence(getLayerDepth());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        resetLayerGradients();
        Sequence nextLayerGradients = getNextLayerGradients();
        Sequence directNextLayerGradients = nextLayerGradients.unjoin(0);
        Sequence reverseNextLayerGradients = nextLayerGradients.unjoin(1);
        if (procedure != null) procedure.calculateGradient(directNextLayerGradients, getLayerGradients(), getTruncateSteps());
        if (reverseProcedure != null) reverseProcedure.calculateGradient(reverseNextLayerGradients, reverseLayerGradients, getTruncateSteps());
        Sequence layerGradients = new Sequence(directNextLayerGradients.getDepth());
        for (Integer sampleIndex : layerGradients.keySet()) {
            layerGradients.put(sampleIndex, getLayerGradients().get(sampleIndex).add(reverseLayerGradients.get(sampleIndex)));
        }
        setLayerGradients(layerGradients);
    }

    /**
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() throws MatrixException {
        HashMap<Matrix, Matrix> layerWeightGradients = new HashMap<>(procedure.getGradients());
        layerWeightGradients.putAll(reverseProcedure.getGradients());
        return layerWeightGradients;
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        HashMap<Matrix, Matrix> weightGradients = reverseProcedure.getGradients();
        for (Matrix weight : weightGradients.keySet()) optimizer.optimize(weight, weightGradients.get(weight));
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        return super.getWeightSet().getNumberOfParameters() + reverseWeightSet.getNumberOfParameters();
    }

}