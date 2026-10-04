def discounted_returns(rewards: list, gamma: float) -> list:
    """
    Returns one discounted return per reward, rounded to four decimals.
    """
    g_t=[]
    
    for t in range(len(rewards)):
        G=0
        for k in range(len(rewards)-t):
            G+=float(rewards[t+k]*gamma**k)
        g_t.append(round(G,4))

    return g_t