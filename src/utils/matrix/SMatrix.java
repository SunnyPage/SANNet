/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Implements sparse matrix.<br>
 * Sparse matrix optimizes matrix memory usage by storing only non-zero values.<br>
 * This matrix type is useful when input sample is expected to contain mostly zero values.<br>
 *
 */
public class SMatrix extends ComputableMatrix {

    /**
     * Matrix data structure as hash map.
     *
     */
    private HashMap<Integer, Double> matrix = new HashMap<>();

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     */
    public SMatrix(double scalarValue) {
        super(1, 1, true);
        updateSliceDimensions(0, 0, 0, 0);
        setValue(0, 0, scalarValue);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public SMatrix(int rows, int columns) {
        super(rows, columns);
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    public SMatrix(int rows, int columns, boolean isScalar) {
        super(rows, columns, isScalar);
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs) {
        this(rows, columns);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, boolean isScalar, Initialization initialization, int inputs, int outputs) {
        this(rows, columns, isScalar);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, Initialization initialization) {
        this(rows, columns);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, boolean isScalar, Initialization initialization) {
        this(rows, columns, isScalar);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, Matrix.Initializer initializer) {
        this(rows, columns);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, boolean isScalar, Matrix.Initializer initializer) {
        this(rows, columns, isScalar);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data) {
        this(rows, columns);
        for (Map.Entry<Integer, Double> entry : data.entrySet()) {
            int index = entry.getKey();
            double value = entry.getValue();
            matrix.put(index, value);
        }
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, boolean isScalar, HashMap<Integer, Double> data) {
        this(rows, columns, isScalar);
        for (Map.Entry<Integer, Double> entry : data.entrySet()) {
            int index = entry.getKey();
            double value = entry.getValue();
            matrix.put(index, value);
        }
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data matrix data.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data, boolean isTransposed) {
        super(rows, columns, false, isTransposed);
        matrix.putAll(data);
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for sparse matrix.
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param data matrix data.
     */
    public SMatrix(int rows, int columns, boolean isScalar, boolean isTransposed, HashMap<Integer, Double> data) {
        super(rows, columns, isScalar, isTransposed);
        matrix.putAll(data);
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        Matrix newMatrix = new SMatrix(getPureRows(), getPureColumns(), isScalar(), false, matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected Matrix applyTranspose() throws MatrixException {
        Matrix newMatrix = new SMatrix(getPureRows(), getPureColumns(), isScalar(), true, matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Returns sub-matrices within matrix.
     *
     * @return sub-matrices within matrix.
     */
    public ArrayList<Matrix> getSubMatrices() {
        ArrayList<Matrix> matrices = new ArrayList<>();
        matrices.add(this);
        return matrices;
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new HashMap<>();
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new SMask(getTotalRows(), getTotalColumns());
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        if (value != 0) matrix.put(isScalar() ? 0 : (getSliceStartRow() + (!isTransposed() ? row : column)) * getPureColumns() + (getSliceStartColumn() + (!isTransposed() ? column : row)), value);
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix.getOrDefault(isScalar() ? 0 : (getSliceStartRow() + (!isTransposed() ? row : column)) * getPureColumns() + (getSliceStartColumn() + (!isTransposed() ? column : row)), (double)0);
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @return new matrix
     */
    protected Matrix getNewMatrix(int rows, int columns) {
        return forceDMatrix ? new DMatrix(rows, columns) : new SMatrix(rows, columns);
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @param isScalar true if matrix is scalar (size 1x1).
     * @return new matrix
     */
    protected Matrix getNewMatrix(int rows, int columns, boolean isScalar) {
        return forceDMatrix ? new DMatrix(rows, columns, isScalar) : new SMatrix(rows, columns, isScalar);
    }

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    protected Matrix getNewMatrix(double constant) {
        return forceDMatrix ? new DMatrix(constant) : new SMatrix(constant);
    }

    /**
     * Return one-hot encoded column vector.
     *
     * @param size size of vector
     * @param position position of one-hot encoded value
     * @return one-hot encoded vector.
     * @throws MatrixException throws exception if position of one-hot encoded value exceeds vector size.
     */
    public static Matrix getOneHotVector(int size, int position) throws MatrixException {
        return getOneHotVector(size, position, true);
    }

    /**
     * Return one-hot encoded vector.
     *
     * @param size size of vector
     * @param position position of one-hot encoded value
     * @param asColumnVector if true one-hot vector is column vector otherwise row vector
     * @return one-hot encoded vector.
     * @throws MatrixException throws exception if position of one-hot encoded value exceeds vector size.
     */
    public static Matrix getOneHotVector(int size, int position, boolean asColumnVector) throws MatrixException {
        if (position > size - 1) throw new MatrixException("Position " + position + " cannot exceed vector size " + size);
        Matrix oneHotVector = new SMatrix(asColumnVector ? size : 1, asColumnVector ? 1 : size);
        oneHotVector.setValue(asColumnVector ? position : 0, asColumnVector ? 0 : position, 1);
        return oneHotVector;
    }

}
