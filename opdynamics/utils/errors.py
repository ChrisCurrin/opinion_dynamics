ECSetupError = RuntimeError(
    """Activities, connection probabilities, social interactions, and dynamics need to be set.
    sn = SocialNetwork(...)
    sn.set_activities(...)
    sn.set_connection_probabilities(...)
    sn.set_social_interactions(...)
    sn.set_dynamics(...)
    sn.run_network(...)
    """
)
