def loss_streak(trades):

    streak = 0

    for r in reversed(trades):

        if r < 0:
            streak += 1
        else:
            break

    return streak


def allow_trade(trades, max_streak=5):

    if loss_streak(trades) >= max_streak:
        return False

    return True