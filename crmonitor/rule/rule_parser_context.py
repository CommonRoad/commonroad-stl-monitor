class RuleParserContext:
    def __init__(self) -> None:
        self._sub_rule_counter = -1

    def generate_new_unique_sub_rule_name(self) -> str:
        """
        Generate a name for a RTAMT sub-rule which is unique in the context of this parse.
        """
        self._sub_rule_counter += 1
        return f"g{self._sub_rule_counter}"
