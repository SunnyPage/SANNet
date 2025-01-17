/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.MatrixException;

/**
 * Implements crosscorrelation matrix operation.
 *
 */
public class CrosscorrelationMatrixOperation extends AbstractConvolutionMatrixOperation {

    /**
     * Constructor for crosscorrelation matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     * @param stride stride step
     */
    public CrosscorrelationMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, filterRowSize, filterColumnSize, dilation, stride);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, double value) throws MatrixException {
        input.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        double resultValue = 0;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                resultValue += input.getValue(filterRow, filterColumn) * filter.getValue(filterRow, filterColumn);
            }
        }
        result.setValue(row, column, resultValue);
        input.unslice();
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMask(int row, int column, double value) throws MatrixException {
        input.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        double resultValue = 0;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                if (!hasMaskAt(filterRow, filterColumn, input, filter)) {
                    resultValue += input.getValue(filterRow, filterColumn) * filter.getValue(filterRow, filterColumn);
                }
            }
        }
        result.setValue(row, column, resultValue);
        input.unslice();
    }

}
