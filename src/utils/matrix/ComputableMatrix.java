/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import utils.matrix.operation.*;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Implements computable operations for matrices.<br>
 *
 */
public abstract class ComputableMatrix extends AbstractMatrix {

    /**
     * If true matrix is treated as scalar (1x1) matrix otherwise as normal matrix.
     *
     */
    private final boolean isScalar;

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride = 1;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation = 1;

    /**
     * Filter row size for convolutional and pooling operations.
     *
     */
    private int filterRowSize;

    /**
     * Filter column size for convolutional and pooling operations.
     *
     */
    private int filterColumnSize;

    /**
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for computable matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    protected ComputableMatrix(int rows, int columns) {
        super(rows, columns);
        this.isScalar = (rows == 1 && columns == 1);
    }

    /**
     * Constructor for computable matrix.<br>
     * If rows and columns do not equal to 1 isScalar parameter is ignored.<br>
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected ComputableMatrix(int rows, int columns, boolean isScalar) {
        super(rows, columns);
        this.isScalar = isScalar && (rows == 1 && columns == 1);
    }

    /**
     * Constructor for computable matrix.
     * If rows and columns do not equal to 1 isScalar parameter is ignored.<br>
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    protected ComputableMatrix(int rows, int columns, boolean isScalar, boolean isTransposed) {
        super(rows, columns, isTransposed);
        this.isScalar = isScalar && (rows == 1 && columns == 1);
    }

    /**
     * Sets parameters for matrix.
     *
     * @param matrix matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected void setParameters(Matrix matrix) throws MatrixException {
        super.setParameters(matrix);
        matrix.setStride(stride);
        matrix.setDilation(dilation);
        matrix.setFilterRowSize(filterRowSize);
        matrix.setFilterColumnSize(filterColumnSize);
    }

    /**
     * Returns value from uniform distribution within -range to +range.
     *
     * @param range range of the distribution.
     * @return random value drawn from the distribution.
     */
    private double uniform(double range) {
        return (2 * random.nextDouble()- 1)  * range;
    }

    /**
     * Returns value from normal distribution defined by standard deviation.
     *
     * @param standardDeviation standard deviation of normal distribution.
     * @return random value drawn from the distribution.
     */
    private double normal(double standardDeviation) {
        return random.nextGaussian() * standardDeviation;
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     */
    public void initialize(Initialization initialization) {
        initialize(initialization, 0, 0);
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public void initialize(Initialization initialization, int inputs, int outputs) {
        switch (initialization) {
            case ZERO -> initialize((Initializer & Serializable) (row, column) -> 0);
            case ONE -> initialize((Initializer & Serializable) (row, column) -> 1);
            case RANDOM -> initialize((Initializer & Serializable) (row, column) -> random.nextDouble());
            case IDENTITY -> initialize((Initializer & Serializable) (row, column) -> (row == column) ? 1 : 0);
            case NORMAL_XAVIER -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (getRows() + getColumns()))));
            case UNIFORM_XAVIER -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows() + getColumns()))));
            case NORMAL_HE -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / ((double) getRows()))));
            case UNIFORM_HE -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows()))));
            case NORMAL_LECUN -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (getRows()))));
            case UNIFORM_LECUN -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (getRows()))));
            case NORMAL_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs + inputs))));
            case UNIFORM_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs + inputs))));
            case NORMAL_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs))));
            case UNIFORM_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs))));
            case NORMAL_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (outputs))));
            case UNIFORM_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (outputs))));
            default -> {
            }
        }
    }

    /**
     * Returns true if matrix is scalar otherwise false.
     *
     * @return true if matrix is scalar otherwise false.
     */
    public boolean isScalar() {
        return isScalar;
    }

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    public void initialize(Matrix.Initializer initializer) {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                setValue(row, column, initializer.value(row, column));
            }
        }
    }

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    public void initializeToValue(double value) {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                setValue(row, column, value);
            }
        }
    }

    /**
     * Increment value of specific row and column.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param value to be added.
     */
    public void incrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) + value);
    }

    /**
     * Decrease value of specific row and column.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param value to be decreased.
     */
    public void decrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) - value);
    }

    /**
     * Multiply value of specific row and column.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param value to be multiplied.
     */
    public void multiplyByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) * value);
    }

    /**
     * Divide value of specific row and column.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param value to be divided.
     */
    public void divideByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) / value);
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        int rows = getRows();
        int columns = getColumns();
        int otherRows = other.getRows();
        int otherColumns = other.getColumns();
        if (otherRows != rows || other.getColumns() != columns) {
            throw new MatrixException("Incompatible target matrix size: " + otherRows + "x" + otherColumns);
        }

        for (int row = 0; row < otherRows; row++) {
            for (int column = 0; column < otherColumns; column++) {
                if (getValue(row, column) != other.getValue(row, column)) return false;
            }
        }
        return true;
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }

        new EqualMatrixOperation(getRows(), getColumns()).apply(other, this);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unary function.
     * @throws MatrixException not thrown in any situation.
     */
    protected void applyFunction(Matrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (result.getRows() != (unaryFunction.getType() != UnaryFunctionType.TRANSPOSE ? getRows() : getColumns()) || result.getColumns() != (unaryFunction.getType() != UnaryFunctionType.TRANSPOSE ? getColumns() : getRows())) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }
        new UnaryMatrixOperation(getRows(), getColumns(), unaryFunction).applyFunction(this, result);
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param binaryFunction binary function.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    protected void applyBiFunction(Matrix other, Matrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (!isScalar() && !other.isScalar() && (getRows() != other.getRows() || getColumns() != other.getColumns())) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (!isScalar() && !result.isScalar() && (getRows() != result.getRows() || getColumns() != result.getColumns())) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }
        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();
        new BinaryMatrixOperation(rows, columns, binaryFunction).applyFunction(this, other, result);
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    protected void applyDot(Matrix other, Matrix result) throws MatrixException {
        if (getColumns() != other.getRows()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (getRows() != result.getRows() || other.getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }
        new DotMatrixOperation(getRows(), other.getColumns()).apply(this, other, result);
    }

    /**
     * Calculates sum of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double sum() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns()).applySum(this);
    }

    /**
     * Calculates mean of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return mean of matrix.
     */
    public double mean() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns()).applyMean(this);
    }

    /**
     * Calculates variance of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return variance of matrix.
     */
    public double variance(double mean) throws MatrixException {
        return new VarianceMatrixOperation(getRows(), getColumns(), mean).applyVariance(this);
    }

    /**
     * Calculates standard deviation of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return standard deviation of matrix.
     */
    public double standardDeviation(double mean) throws MatrixException {
        return new VarianceMatrixOperation(getRows(), getColumns(), mean).applyStandardDeviation(this);
    }

    /**
     * Calculates cumulative p- norm (p is number equal or bigger than 1) of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return norm of matrix.
     */
    public double norm(int p) throws MatrixException {
        return new NormMatrixOperation(getRows(), getColumns(), p).apply(this);
    }

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return normalized matrix.
     */
    public Matrix normalize(boolean inplace) throws MatrixException {
        return new NormalizeMatrixOperation(getRows(), getColumns(), mean(), variance()).apply(this, inplace ? this : getNewMatrix());
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return minimum value of matrix.
     */
    public double min() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns()).applyMin(this);
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to minimum value of matrix.
     */
    public int[] argmin() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns()).applyArgMin(this);
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return maximum value of matrix.
     */
    public double max() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns()).applyMax(this);
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to maximum value of matrix.
     */
    public int[] argmax() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns()).applyArgMax(this);
    }

    /**
     * Calculates entropy of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double entropy() throws MatrixException {
        return new EntropyMatrixOperation(getRows(), getColumns()).applyEntropy(this);
    }

    /**
     * Returns softmax of matrix.
     *
     * @param result result matrix.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        final double maxValue = max();
        applyFunction(result, new UnaryFunction(value -> Math.exp(value - maxValue)));
        result.divide(result.sum(), result);
        return result;
    }

    /**
     * Returns Gumbel softmax of matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result, double gumbelSoftmaxTau) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        // https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
        applyFunction(result, new UnaryFunction(value -> (value + getGumbelNoise()) / gumbelSoftmaxTau));
        final double maxValue = max();
        applyFunction(result, new UnaryFunction(value -> Math.exp(value - maxValue)));
        result.divide(result.sum(), result);
        return result;
    }

    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-20;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

    /**
     * Returns softmax gradient of matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getRows() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }
        return new SoftmaxGradientMatrixOperation(getRows(), getRows()).apply(this, result);
    }

    /**
     * Sets stride size for convolution and pooling operations.
     *
     * @param stride stride size.
     */
    public void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Returns stride size for convolution and pooling operations.
     *
     * @return stride size.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Sets dilation step size for convolution operations.
     *
     * @param dilation dilation step size.
     */
    public void setDilation(int dilation) {
        this.dilation = dilation;
    }

    /**
     * Returns dilation step size for convolution operations.
     *
     * @return dilation step size.
     */
    public int getDilation() {
        return dilation;
    }

    /**
     * Sets filter row size for convolution and pooling operations.
     *
     * @param filterRowSize filter row size.
     */
    public void setFilterRowSize(int filterRowSize) {
        this.filterRowSize = filterRowSize;
    }

    /**
     * Sets filter column size for convolution and pooling operations.
     *
     * @param filterColumnSize filter column size.
     */
    public void setFilterColumnSize(int filterColumnSize) {
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Returns filter row size for convolution and pooling operations.
     *
     * @return filter row size for convolution and pooling operations.
     */
    public int getFilterRowSize() {
        return filterRowSize;
    }

    /**
     * Returns filter column size for convolution and pooling operations.
     *
     * @return filter column size for convolution and pooling operations.
     */
    public int getFilterColumnSize() {
        return filterColumnSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyConvolve(Matrix filter, Matrix result) throws MatrixException {
        new ConvolutionMatrixOperation(result.getRows(), result.getColumns(), filter.getRows(), filter.getColumns(), getDilation(), getStride()).apply(this, filter, result);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyCrosscorrelate(Matrix filter, Matrix result) throws MatrixException {
        new CrosscorrelationMatrixOperation(result.getRows(), result.getColumns(), filter.getRows(), filter.getColumns(), getDilation(), getStride()).apply(this, filter, result);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyWinogradConvolve(Matrix filter, Matrix result) throws MatrixException {
        new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns()).apply(this, filter, result);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyWinogradConvolve(Matrix filter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), A, AT, C, CT, G, GT).apply(this, filter, result);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyWinogradConvolve(Matrix preprocessedFilter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), A, AT, C, CT).apply(this, preprocessedFilter, result);
    }

    /**
     * Calculates gradient of convolution for input.
     *
     * @param filter filter for convolution operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputGradient input gradient.
     */
    public void convolveInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException {
        new ConvolutionInputGradientMatrixOperation(getRows(), getColumns(), filter.getRows(), filter.getColumns(), getDilation(), getStride()).apply(this, filter, inputGradient);
    }

    /**
     * Calculates gradient of crosscorrelation for input.
     *
     * @param filter filter for crosscorrelation operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputGradient input gradient.
     */
    public void crosscorrelateInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException {
        new CrosscorrelationInputGradientMatrixOperation(getRows(), getColumns(), filter.getRows(), filter.getColumns(), getDilation(), getStride()).apply(this, filter, inputGradient);
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param filterGradient filter gradient.
     */
    public void convolveFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException {
        new ConvolutionFilterGradientMatrixOperation(getRows(), getColumns(), filterGradient.getRows(), filterGradient.getColumns(), getDilation(), getStride()).apply(this, input, filterGradient);
    }

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param filterGradient filter gradient.
     */
    public void crosscorrelateFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException {
        new CrosscorrelationFilterGradientMatrixOperation(getRows(), getColumns(), filterGradient.getRows(), filterGradient.getColumns(), getDilation(), getStride()).apply(this, input, filterGradient);
    }

    /**
     * Calculates max pooling operation for matrix and returns max arguments.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param maxPos maximum position for each result row and column value.
     */
    protected void applyMaxPool(Matrix result, HashMap<Integer, Integer> maxPos) throws MatrixException {
        new MaxPoolMatrixOperation(result.getRows(), result.getColumns(), getColumns(), getFilterRowSize(), getFilterColumnSize(), getStride()).apply(this, maxPos, result);
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param maxPos maximum position for each result row and column value.
     */
    public void maxPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> maxPos) throws MatrixException {
        new MaxPoolGradientMatrixOperation(getRows(), getColumns(), inputGradient.getColumns(), getStride()).apply(this, maxPos, inputGradient);
    }

    /**
     * Calculates random pooling operation for matrix and returns input positions.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputPos input position for each result row and column value.
     */
    protected void applyRandomPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException {
        new RandomPoolMatrixOperation(result.getRows(), result.getColumns(), getColumns(), getFilterRowSize(), getFilterColumnSize(), getStride()).apply(this, inputPos, result);
    }

    /**
     * Calculates gradient for random pool operation.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputPos input position for each result row and column value.
     */
    public void randomPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> inputPos) throws MatrixException {
        new RandomPoolGradientMatrixOperation(getRows(), getColumns(), inputGradient.getColumns(), getStride()).apply(this, inputPos, inputGradient);
    }

    /**
     * Calculates cyclic pooling operation for matrix and returns input positions.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputPos input position for each result row and column value.
     */
    protected void applyCyclicPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException {
        new CyclicPoolMatrixOperation(result.getRows(), result.getColumns(), getColumns(), getFilterRowSize(), getFilterColumnSize(), getStride()).apply(this, inputPos, result);
    }

    /**
     * Calculates gradient for cyclic pool operation.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputPos input position for each result row and column value.
     */
    public void cyclicPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> inputPos) throws MatrixException {
        new CyclicPoolGradientMatrixOperation(getRows(), getColumns(), inputGradient.getColumns(), getStride()).apply(this, inputPos, inputGradient);
    }

    /**
     * Calculates average pooling operation for matrix.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyAveragePool(Matrix result) throws MatrixException {
        new AveragePoolMatrixOperation(result.getRows(), result.getColumns(), getFilterRowSize(), getFilterColumnSize(), getStride()).apply(this, result);
    }

    /**
     * Calculates gradient of average pooling operation for matrix.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void averagePoolGradient(Matrix inputGradient) throws MatrixException {
        new AveragePoolGradientMatrixOperation(getRows(), getColumns(), getFilterRowSize(), getFilterColumnSize(), getStride()).apply(this, inputGradient);
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify() throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns()).apply(this, getNewMatrix());
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify(double multiLabelThreshold) throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns(), multiLabelThreshold).apply(this, getNewMatrix());
    }

    /**
     * Encodes to value to column row vector.
     *
     * @param value value
     * @return bit column vector.
     * @throws MatrixException throws exception if binary code size is exceeding number of maximum bits.
     */
    public static Matrix encodeToBitColumnVector(int value) throws MatrixException {
        return ComputableMatrix.encodeToBitColumnVector(value, value);
    }

    /**
     * Encodes to value to bit column vector.
     *
     * @param value value
     * @param maxBits max number of bits.
     * @return bit column vector.
     * @throws MatrixException throws exception if binary code size is exceeding number of maximum bits.
     */
    public static Matrix encodeToBitColumnVector(int value, int maxBits) throws MatrixException {
        String binaryCode = String.format("%" + maxBits + "s", Integer.toBinaryString(value)).replaceAll(" ", "0");
        if (binaryCode.length() > maxBits) throw new MatrixException("Binary code: '" + binaryCode + "' has length " + binaryCode.length() + " exceeding number of maximum bits " + maxBits);
        Matrix encodedMatrix = new SMatrix(binaryCode.length(), 1);
        int binaryCodeLength = binaryCode.length();
        for (int charIndex = 0; charIndex < binaryCodeLength; charIndex++) {
            char charAt = binaryCode.charAt(charIndex);
            if (charAt == '1') encodedMatrix.setValue(charIndex, 0, 1);
        }
        return encodedMatrix;
    }

    /**
     * Encodes bit column vector value
     *
     * @return value
     * @throws MatrixException throws exception if matrix is not bit column vector.
     */
    public int encodeToValue() throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be column vector.");
        int rows = getRows();
        int result = 0;
        for (int row = 0; row < rows; row++) {
            double value = getValue(row, 0);
            if (!(value == 0 || value == 1)) throw new MatrixException("Bit column vector must contains values of 0 or 1.");
            result += value * Math.pow(2, (rows - 1) - row);
        }
        return result;
    }

    /**
     * Returns number of bits needed to represent value.
     *
     * @param value value.
     * @return number of bits needed to represent value.
     */
    public static int numberOfBits(int value) {
        return (int)Math.floor((Math.log10(value) / Math.log10(2) + 1));
    }

}
