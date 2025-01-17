/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Implements abstract mask that implements common operations for masking.<br>
 *
 */
public abstract class AbstractMask implements Cloneable, Serializable, Mask {

    @Serial
    private static final long serialVersionUID = -4902569287054022460L;

    /**
     * Defines number of rows in mask.
     *
     */
    private final int rows;

    /**
     * Defines number of columns in mask.
     *
     */
    private final int columns;

    /**
     * Bernoulli-probability for selecting if entry (row, column) is masked or not.
     *
     */
    private double probability = 0;

    /**
     * If true mask is transposed.
     *
     */
    private boolean isTransposed = false;

    /**
     * Random function for mask class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for abstract mask.
     *
     * @param rows number of rows in mask.
     * @param columns number of columns in mask.
     */
    public AbstractMask(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }

    /**
     * Constructor for abstract mask.
     *
     * @param rows number of rows in mask.
     * @param columns number of columns in mask.
     * @param probability probability of masking.
     * @param isTransposed is true mask is transposed otherwise false.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public AbstractMask(int rows, int columns, double probability, boolean isTransposed) throws MatrixException {
        this.rows = rows;
        this.columns = columns;
        setProbability(probability);
        this.isTransposed = isTransposed;
    }

    /**
     * Returns size (rows * columns) of mask
     *
     * @return size of mask.
     */
    public int size() {
        return rows * columns;
    }

    /**
     * Returns number of rows in mask.
     *
     * @return number of rows in mask.
     */
    public int getRows() {
        return !isTransposed() ? rows : columns;
    }

    /**
     * Returns number of columns in mask.
     *
     * @return number of columns in mask.
     */
    public int getColumns() {
        return !isTransposed() ? columns : rows;
    }

    /**
     * Creates new mask with object reference to the mask data of this mask.
     *
     * @return newly created reference mask.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    public Mask reference() throws MatrixException {
        Mask newMask;
        // Make shallow copy of mask leaving references internal objects which are shared.
        try {
            newMask = (Mask)super.clone();
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of mask failed.");
        }
        return newMask;
    }

    /**
     * Creates new mask with full copy of this mask.
     *
     * @return newly created mask copy.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public Mask copy() throws MatrixException {
        return getCopy();
    }

    /**
     * Transposes mask.
     *
     * @return reference to this mask but as transposed with flipped rows and columns.
     */
    public Mask transpose() throws MatrixException {
        Mask transposedMask = reference();
        transposedMask.setTranspose(true);
        return transposedMask;
    }

    /**
     * Checks if mask is transposed.
     *
     * @return true is mask is transposed otherwise false.
     */
    public boolean isTransposed() {
        return isTransposed;
    }

    /**
     * Sets if mask is transposed.
     *
     * @param isTransposed if true mask is transposed and if false not transposed.
     */
    public void setTranspose(boolean isTransposed) {
        this.isTransposed = isTransposed;
    }

    /**
     * Checks if mask is set at specific row and column
     *
     * @param row row.
     * @param column column.
     * @return return true if mask is set at row and column.
     */
    public boolean isMasked(int row, int column) {
        return getMask(row, column);
    }

    /**
     * Sets bernoulli probability to mask specific row and column.
     *
     * @param probability masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public void setProbability(double probability) throws MatrixException {
        if (probability < 0 || probability > 1) throw new MatrixException("Masking probability must be between 0 and 1.");
        this.probability = probability;
    }

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    public double getProbability() {
        return probability;
    }

    /**
     * Returns true with defined masking probability.
     *
     * @return true with defined masking probability.
     */
    private boolean isMaskedByProbability() {
        return random.nextDouble() > probability;
    }

    /**
     * Sets masking with given bernoulli probability for each row and column.
     *
     */
    public void maskByProbability() {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                setMask(row, column, isMaskedByProbability());
            }
        }
    }

    /**
     * Sets row masking for this mask with given bernoulli probability.
     *
     */
    public void maskRowByProbability() {
        int rows = getRows();
        for (int row = 0; row < rows; row++) {
            setRowMask(row, isMaskedByProbability());
        }
    }

    /**
     * Sets column masking for this mask with given bernoulli probability.
     *
     */
    public void maskColumnByProbability() {
        int columns = getColumns();
        for (int column = 0; column < columns; column++) {
            setColumnMask(column, isMaskedByProbability());
        }
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        int columns = getColumns();
        for (int column = 0; column < columns; column++) setMask(row, column, value);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        int rows = getRows();
        for (int row = 0; row < rows; row++)  setMask(row, column, value);
    }

    /**
     * Prints mask in row and column format.
     *
     */
    public void print() {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            System.out.print("[");
            for (int column = 0; column < columns; column++) {
                System.out.print((isMasked(row, column) ? 1 : 0));
                if (column < columns - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

    /**
     * Prints size (rows x columns) of mask.
     *
     */
    public void printSize() {
        System.out.println("Mask size: " + getRows() + "x" + getColumns());
    }

}
