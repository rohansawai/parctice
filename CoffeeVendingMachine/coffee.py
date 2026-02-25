from abc import ABC, abstractmethod
from typing import Dict
from enums import Ingredient

class Coffee(ABC):
    def __init__(self):
        self.coffee_type = "Unknown Coffee"

    def get_coffee_type(self) -> str:
        return self.coffee_type
    
    def prepare(self):
        print(f"\nPreparing your {self.get_coffee_type()}...")
        self._grind_beans()
        self._brew()
        self.add_condiments()
        self._pour_into_cup()
        print(f"{self.get_coffee_type()} is ready!")

    def _grind_beans(self):
        pass

    def _brew(self):
        pass

    def _pour_into_cup(self):
        pass

    @abstractmethod
    def add_condiments(self):
        pass

    @abstractmethod
    def get_price(self) -> int:
        pass

    @abstractmethod
    def get_recipe(self) -> Dict[Ingredient, int]:
        pass


class Espresso(Coffee):
    def __init__(self):
        super().__init__()
        self.coffee_type = "Espresso"

    def add_condiments(self):
        pass

    def get_price(self) -> int:
        return 150
    
    def get_recipe(self) -> Dict[Ingredient, int]:
        return {Ingredient.COFFEE_BEANS: 7, Ingredient.WATER: 30}
    

class Cappuccino(Coffee):
    