/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Implements max pooling gradient matrix operation.
 *
 */
public class MaxPoolGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    private Matrix outputGradient;

    /**
     * Input gradient.
     *
     */
    private Matrix inputGradient;

    /**
     * Number of inputs columns.
     *
     */
    private final int inputColumnSize;

    /**
     * Maximum position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> maxPos;

    /**
     * Constructor for max pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     * @param stride stride step
     */
    public MaxPoolGradientMatrixOperation(int rows, int columns, int inputColumnSize, int stride) {
        super(rows, columns, true, stride);
        this.inputColumnSize = inputColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param maxPos maximum positions.
     * @param inputGradient input gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */

    public Matrix apply(Matrix outputGradient, HashMap<Integer, Integer> maxPos, Matrix inputGradient) throws MatrixException {
        this.outputGradient = outputGradient;
        this.maxPos = maxPos;
        this.inputGradient = inputGradient;
        applyMatrixOperation();
        return inputGradient;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return outputGradient;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getAnother() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        inputGradient.setValue(maxPos.get(2 * (row * inputColumnSize + column)), maxPos.get(2 * (row * inputColumnSize + column) + 1), value);
    }

}
