class RiskManager:
    def __init__(self, kelly_fraction=0.5, martingale_factor=2):
        self.kelly_fraction = kelly_fraction
        self.martingale_factor = martingale_factor
        self.last_trade_was_loss = False
        self.last_bet = 0

    def get_position_size(self, proba, account):
        kelly = max(0, proba - (1 - proba))
        base_bet = self.kelly_fraction * kelly * account.capital
        if self.last_trade_was_loss:
            bet = min(base_bet * self.martingale_factor, account.capital)
        else:
            bet = base_bet
        self.last_bet = bet
        return bet

    def update_after_trade(self, pnl):
        self.last_trade_was_loss = pnl < 0 