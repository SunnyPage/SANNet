/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableProximalPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Proximal Policy Optimization (PPO) algorithm.<br>
 *
 */
public class PPO extends AbstractPolicyGradient {

    /**
     * Constructor for Proximal Policy Optimization
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public PPO(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws ClassNotFoundException, DynamicParamException, IOException, AgentException, MatrixException {
        super(environment, new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for Proximal Policy Optimization
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public PPO(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        super(environment, new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public PPO reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference();
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new PPO(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public PPO reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference(sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new PPO(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
    }

}