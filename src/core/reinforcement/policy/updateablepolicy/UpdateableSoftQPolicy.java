/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.optimization.Adam;
import core.optimization.Optimizer;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements updateable soft Q policy.<br>
 *
 */
public class UpdateableSoftQPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for updateable soft Q policy.
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(softQAlpha:DOUBLE), " +
            "(autoSoftAlpha:BOOLEAN)";

    /**
     * Alpha parameter for entropy control.
     *
     */
    private double softQAlpha;

    /**
     * If true alpha is adjusted automatically otherwise not.
     *
     */
    private boolean autoSoftAlpha;

    /**
     * Alpha parameter for entropy in matrix form.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * Cumulative alpha loss gradient.
     *
     */
    private transient double alphaLossGradient = 0;

    /**
     * Update count for cumulative alpha loss gradient.
     *
     */
    private transient int alphaLossGradientCount = 0;

    /**
     * Optimizer for alpha loss.
     *
     */
    private final Optimizer optimizer = new Adam("learningRate = 0.001");

    /**
     * Constructor for updateable soft Q policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
        this.softQAlphaMatrix = softQAlphaMatrix;
        this.softQAlphaMatrix.setValue(0, 0, softQAlpha);
    }

    /**
     * Constructor for updateable soft Q policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @param params parameters for updateable soft Q policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
        this.softQAlphaMatrix = softQAlphaMatrix;
        this.softQAlphaMatrix.setValue(0, 0, softQAlpha);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        autoSoftAlpha = true;
        softQAlpha = 0.25;
    }

    /**
     * Returns parameters used for updateable soft Q policy.
     *
     * @return parameters used for updateable soft Q policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableSoftQPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for updateable soft Q policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *
     * @param params parameters used for updateable soft Q policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("softQAlpha")) softQAlpha = params.getValueAsDouble("softQAlpha");
        if (params.hasParam("autoSoftAlpha")) autoSoftAlpha = params.getValueAsBoolean("autoSoftAlpha");
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), functionEstimator.reference(), new DMatrix(0), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), new DMatrix(0), params);
    }

    /**
     * Returns reference to policy function.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used between policy functions otherwise separate policy function estimator is used.
     * @param memory reference to memory.
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(memory), new DMatrix(0), params);
    }

    /**
     * Returns soft alpha.
     *
     * @return soft alpha.
     */
    protected double getSoftAlpha() {
        return softQAlpha;
    }

    /**
     * Returns true if alpha is adjusted automatically otherwise not.
     *
     * @return true if alpha is adjusted automatically otherwise not.
     */
    protected boolean isAutoSoftAlpha() {
        return autoSoftAlpha;
    }

    /**
     * Updates function estimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        super.updateFunctionEstimator();
        if (isAutoSoftAlpha()) updateAlpha();
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition);
        double currentPolicyValue = currentPolicyValues.getValue(stateTransition.action, 0);
        if (isAutoSoftAlpha()) incrementPolicyValues(currentPolicyValues, stateTransition.action, stateTransition.environmentState.availableActions().size());
        return getSoftAlpha() * Math.log(currentPolicyValue) - stateTransition.tdTarget;
    }

    /**
     * Increments policy values.
     *
     * @param policyValues policy values.
     * @param action action.
     */
    private void incrementPolicyValues(Matrix policyValues, int action, int actionsAvailable) {
        alphaLossGradient += -Math.log(policyValues.getValue(action, 0)) - 0.98 * Math.log(actionsAvailable);
        alphaLossGradientCount++;
    }

    /**
     * Updates alpha.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void updateAlpha() throws MatrixException, DynamicParamException {
        // https://raw.githubusercontent.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/master/ContinousControl/SAC.ipynb
        // self.target_entropy = -action_size  # -dim(A)
        // alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        softQAlphaMatrix.setValue(0,0, softQAlpha);
        optimizer.optimize(softQAlphaMatrix, new DMatrix(-alphaLossGradient / (double)alphaLossGradientCount));
        softQAlpha = softQAlphaMatrix.getValue(0,0);
        if (++printCount == 25) {
            System.out.println("Soft Q alpha: " + softQAlpha);
            printCount = 0;
        }
        alphaLossGradientCount = 0;
        alphaLossGradient = 0;
    }

    private int printCount = 0;

}
