/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.StateTransition;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyFactory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;

/**
 * Implements abstract policy with common policy functions.<br>
 *
 */
public abstract class AbstractPolicy implements Policy, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = 7604226764648819354L;

    /**
     * Reference to function estimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * Reference to executable policy.
     *
     */
    protected final ExecutablePolicy executablePolicy;

    /**
     * If true agent is in learning mode.
     *
     */
    private transient boolean isLearning = true;

    /**
     * Parameters for policy.
     *
     */
    protected final String params;

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        this(executablePolicyType, functionEstimator, null);
    }

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        this(ExecutablePolicyFactory.create(executablePolicyType), functionEstimator, params);
    }

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) throws AgentException, DynamicParamException {
        this(executablePolicy, functionEstimator, null);
    }

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        initializeDefaultParams();
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        if (getFunctionEstimator().isStateActionValueFunction() && !isUpdateablePolicy()) throw new AgentException("Non-updateable policy cannot be applied along state value function estimator.");
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for abstract policy.
     *
     * @return parameters used for abstract policy.
     */
    public String getParamDefs() {
        return getExecutablePolicy().getParamDefs() + ", " + getFunctionEstimator().getParamDefs();
    }

    /**
     * Sets parameters used for abstract policy.<br>
     *
     * @param params parameters used for abstract policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        getExecutablePolicy().setParams(params);
        getFunctionEstimator().setParams(params);
    }

    /**
     * Starts abstract policy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        getFunctionEstimator().start();
    }

    /**
     * Stops abstract policy.
     *
     */
    public void stop() {
        getFunctionEstimator().stop();
    }

    /**
     * Registers agent for function estimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        getFunctionEstimator().registerAgent(agent);
    }

    /**
     * Returns executable policy.
     *
     * @return executable policy.
     */
    public ExecutablePolicy getExecutablePolicy() {
        return executablePolicy;
    }

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    public void setLearning(boolean isLearning) {
        this.isLearning = isLearning;
        getExecutablePolicy().setLearning(isLearning);
    }

    /**
     * Return flag is policy is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    public boolean isLearning() {
        return isLearning;
    }

    /**
     * Increments executable policy.
     *
     */
    public void increment() {
        getExecutablePolicy().increment();
    }

    /**
     * Returns values for state.
     *
     * @param functionEstimator function estimator.
     * @param stateTransition state.
     * @param isAction true if prediction is for taking other otherwise false.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getValues(FunctionEstimator functionEstimator, StateTransition stateTransition, boolean isAction) throws MatrixException, NeuralNetworkException {
        return functionEstimator.predictPolicyValues(stateTransition, isAction);
    }

    /**
     * Takes action defined by external agent.
     * @param stateTransition state transition.
     *
     */
    public void act(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        getExecutablePolicy().action(getValues(getFunctionEstimator(), stateTransition, true), stateTransition.environmentState.availableActions(), stateTransition.action);
    }

    /**
     * Takes action defined by executable policy.
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        stateTransition.action = getExecutablePolicy().action(getValues(getFunctionEstimator(), stateTransition, true), stateTransition.environmentState.availableActions(), !isLearning() || alwaysGreedy);
        if (isLearning()) getFunctionEstimator().add(stateTransition);
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to function estimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

}
