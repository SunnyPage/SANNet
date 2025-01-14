/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

/**
 * Implements abstract unary expression.<br>
 *
 */
public abstract class AbstractUnaryExpression extends AbstractExpression {

    /**
     * Constructor for abstract unary expression.
     *
     * @param name name of expression.
     * @param operationSignature operation signature
     * @param expressionID expression ID
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractUnaryExpression(String name, String operationSignature, int expressionID, Node argument1, Node result) throws MatrixException {
        super(name, operationSignature, expressionID, argument1, result);
    }

}
