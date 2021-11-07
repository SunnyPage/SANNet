/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

/**
 * Implements L2 (ridge) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L2_Regularization extends AbstractRegularization {

    /**
     * Parameter name types for L2_Regularization.
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(lambda:DOUBLE)";

    /**
     * Regularization rate.
     *
     */
    private double lambda;

    /**
     * Constructor for L2 regularization class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L2_Regularization() throws DynamicParamException {
        super(RegularizationType.L2_REGULARIZATION, L2_Regularization.paramNameTypes);
    }

    /**
     * Constructor for L2 regularization class.
     *
     * @param params parameters for L2 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L2_Regularization(String params) throws DynamicParamException {
        super(RegularizationType.L2_REGULARIZATION, L2_Regularization.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        lambda = 0.01;
    }

    /**
     * Sets parameters used for L2 regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     * @param params parameters used for L2 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
    }

    /**
     * Not used.
     *
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {}

    /**
     * Not used.
     *
     * @param inputs inputs.
     */
    public void forward(MMatrix inputs) {}

    /**
     * Calculates and returns cumulated error from L2 regularization.<br>
     * This is added to the total output error of neural network.<br>
     *
     * @param weight weight matrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from L2 regularization.
     */
    public double error(Matrix weight) throws DynamicParamException, MatrixException {
        return lambda * weight.power(2).sum();
    }

    /**
     * Regulates weights by calculating 2- norm of weights and adding it to weight gradient sum.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        weightGradientSum.add(weight.multiply(2 * lambda), weightGradientSum);
    }

}

