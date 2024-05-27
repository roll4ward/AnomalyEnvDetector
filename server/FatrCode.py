class FatrCode:
    """
    This class contains various codes and their meanings used in the application.
    """

    CO: str = "CO2변화량"  # CO2 Change Amount
    EC: str = "탄소변화량"  # Carbon Change Amount
    CN: str = "탄소분석결과"  # Carbon Analysis Result
    AB: str = "흡수탄소량"  # Absorbed Carbon Amount
    EM: str = "배출탄소량"  # Emitted Carbon Amount

    def __init__(self):
        """
        The constructor for FatrCode class.
        """
        pass

    @staticmethod
    def get_code_meaning(code: str) -> str:
        """
        Get the meaning of a given code.
        
        Parameters:
        code (str): The code to look up.
        
        Returns:
        str: The meaning of the code.
        """
        code_meanings = {
            'CO': "CO2변화량 (CO2 Change Amount)",
            'EC': "탄소변화량 (Carbon Change Amount)",
            'CN': "탄소분석결과 (Carbon Analysis Result)",
            'AB': "흡수탄소량 (Absorbed Carbon Amount)",
            'EM': "배출탄소량 (Emitted Carbon Amount)"
        }
        return code_meanings.get(code, "Unknown Code")
    
    @classmethod
    def as_dict(cls) -> dict:
        return {key: value for key, value in vars(cls).items() if not key.startswith('__') and not callable(getattr(cls, key))}