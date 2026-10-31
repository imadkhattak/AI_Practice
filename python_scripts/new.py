class Robot:

    def __init__(self, name, battery=100):
        self.name = name
        self.battery = battery


    def recharge(self, amount):

        self.battery = min(100, self.battery + amount)
        print(f"{self.name} battery recharged to {self.battery}%")

bot = Robot("Atlas")

# These two lines are 100% identical in Python:
bot.recharge(25)         # Normal syntax
Robot.recharge(bot, 25)  # What Python actually does!

# Inspect instance state dictionary
print(bot.__dict__)
# Output: {'name': 'Atlas', 'battery': 100}
