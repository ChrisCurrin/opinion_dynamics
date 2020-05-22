ECSetupError = RuntimeError(
    """Activities, connection probabilities, social interactions, and dynamics need to be set.
    ec = EchoChamber(...)
    ec.set_activities(...)
    ec.set_connection_probabilities(...)
    ec.set_social_interactions(...)
    ec.set_dynamics(...)
    ec.run_network(...)
    """
)
