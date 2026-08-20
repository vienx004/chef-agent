class CulinaryConverter:
    """
    A utility class providing conversions for common culinary measurements.
    Handles weights, volumes (with density approximations for flour, sugar, butter),
    and temperatures.
    """

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Converts Fahrenheit to Celsius, rounded to 1 decimal place."""
        return round((fahrenheit - 32) * 5 / 9, 1)

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Converts Celsius to Fahrenheit, rounded to 1 decimal place."""
        return round((celsius * 9 / 5) + 32, 1)

    @staticmethod
    def ounces_to_grams(ounces: float) -> float:
        """Converts ounces to grams, rounded to 1 decimal place."""
        return round(ounces * 28.3495, 1)

    @staticmethod
    def grams_to_ounces(grams: float) -> float:
        """Converts grams to ounces, rounded to 1 decimal place."""
        return round(grams / 28.3495, 1)

    @staticmethod
    def cups_to_grams(cups: float, ingredient: str = "liquid") -> float:
        """
        Converts cups to grams based on standard culinary density approximations.

        Args:
            cups (float): Number of cups.
            ingredient (str): Type of ingredient (e.g. 'flour', 'sugar', 'butter', 'liquid').

        Returns:
            float: Gram weight equivalent.
        """
        ing = ingredient.lower().strip()
        
        if "flour" in ing:
            # Standard all-purpose flour: 1 cup ≈ 120g
            return round(cups * 120.0, 1)
        elif "sugar" in ing:
            if "brown" in ing:
                # Packed brown sugar: 1 cup ≈ 200g
                return round(cups * 200.0, 1)
            elif "powdered" in ing or "confectioners" in ing:
                # Powdered sugar: 1 cup ≈ 120g
                return round(cups * 120.0, 1)
            else:
                # Granulated sugar: 1 cup ≈ 200g
                return round(cups * 200.0, 1)
        elif "butter" in ing:
            # Unsalted/salted butter: 1 cup = 2 sticks ≈ 227g
            return round(cups * 227.0, 1)
        elif "cocoa" in ing:
            # Unsweetened cocoa powder: 1 cup ≈ 100g
            return round(cups * 100.0, 1)
        else:
            # Default to water-like liquids: 1 cup ≈ 236.6g
            return round(cups * 236.6, 1)
