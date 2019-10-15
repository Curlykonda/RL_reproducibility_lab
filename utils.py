def get_epsilon(it, epoch, gamma=0.99):
    #return 1 - min((it/50000)*.95, .95)
    return 0.3 * (gamma**epoch)



