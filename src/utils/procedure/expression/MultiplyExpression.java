/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.BinaryFunction;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements expression for multiply operation.<br>
 *
 */
public class MultiplyExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Reference to multiply matrix operation.
     *
     */
    private final BinaryMatrixOperation multiplyMatrixOperation;

    /**
     * Constructor for multiply operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MultiplyExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("MULTIPLY", "*", expressionID, argument1, argument2, result);

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !argument1.isScalar() ? argument1.getRows() : argument2.getRows();
        int columns = !argument1.isScalar() ? argument1.getColumns() : argument2.getColumns();

        multiplyMatrixOperation = new BinaryMatrixOperation(rows, columns, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2));
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return false;
    }

    /**
     * Calculates expression.
     *
     */
    public void calculateExpression() {
    }

    /**
     * Calculates expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        checkArguments(argument1, argument2, sampleIndex);
        multiplyMatrixOperation.applyFunction(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex), result.getNewMatrix(sampleIndex));
    }

    /**
     * Calculates gradient of expression.
     *
     */
    public void calculateGradient() {
    }

    /**
     * Calculates gradient of expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        checkResultGradient(result, sampleIndex);
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, multiplyMatrixOperation.applyFunction(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex), argument1.getNewMatrix()), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, multiplyMatrixOperation.applyFunction(argument1.getMatrix(sampleIndex), result.getGradient(sampleIndex), argument2.getNewMatrix()), false);

    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printBasicBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " " + getOperationSignature() + " " + argument2.getName());
        printArgument2Gradient(true, false, " " + getOperationSignature() + " " + argument1.getName());
    }

}
