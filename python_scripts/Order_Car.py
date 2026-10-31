class Revo:

    def __init__(self, color="White", speed=180, engine="2.8L Diesel"):
        self.color = color
        self.speed = speed
        self.engine = engine

    def show_color(self):
        print("Color:", self.color)

    def show_speed(self):
        print("Speed:", self.speed, "km/h")

    def show_engine(self):
        print("Engine:", self.engine)


# Person 1 orders a Revo
person1_car = Revo()

person1_car.show_color()
person1_car.show_speed()
person1_car.show_engine()


# Person 2 orders a Revo with different specifications
person2_car = Revo("Black", 200, "2.8L Diesel")

person2_car.show_color()
person2_car.show_speed()
person2_car.show_engine()