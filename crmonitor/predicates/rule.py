import inspect
import re
import sys
from enum import auto, Enum


# from crmonitor.common.evaluation import bool_to_norm_rob
# from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor


def get_all_predicate_evaluators():
    mod_name = "crmonitor.predicates.predicate"
    # noinspection PyUnresolvedReferences
    import crmonitor.predicates.predicate
    classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
    classes = list(filter(lambda p: "Pred" in p[0], classes))
    d = {}
    for name, cls in classes:
        d[cls.predicate_name] = cls
    return d


class IOType(Enum):
    OUTPUT = auto()
    INPUT = auto()


class QuantificationType(Enum):
    ALL = auto()
    EXISTENTIAL = auto()
    NONE = auto()


def parse_rule(full_rule_str, config, name=None):
    full_predicate_pattern = re.compile(r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)?_(?P<agents>(_a(\d)+)+)))")
    quantification_pattern = re.compile(r"^(?P<quant>[AE])\sa(?P<veh_id>\d+):\s\((?P<rule>.*)\)$")
    subrule_pattern = re.compile(r"[AE]\sa\d+:\s\(.*\)")
    if name is None:
        name = full_rule_str
    m = quantification_pattern.match(full_rule_str)
    if m is not None:
        # Quantification on top level
        # mod_rule_str = mod_rule_str[
        #                :m.start()] + f"g{len(sub_rule_str)}" + mod_rule_str[
        #                                                        m.end():]
        sub_rule_str = m["rule"]
        children = parse_rule(sub_rule_str, config, "g0")
        quantified_vehicle = int(m.group("veh_id"))
        if m.group("quant") == "E":
            node = ExistNode([children], quantified_vehicle, name)
        elif m.group("quant") == "A":
            node = AllNode([children], quantified_vehicle, name)
        else:
            raise ValueError()
    else:
        mod_rule_str = full_rule_str
        predicate_assignment = set()
        sub_rules = []
        m = subrule_pattern.match(mod_rule_str)
        while m is not None:
            mod_rule_str = mod_rule_str[
                           :m.start()] + f"g{len(sub_rules)}" + mod_rule_str[
                                                                   m.end():]
            sub_rule_str = m[0]
            sub_rules.append(
                parse_rule(sub_rule_str, config, f"g{len(sub_rules)}"))
            m = subrule_pattern.match(mod_rule_str)

        pred_matches = full_predicate_pattern.finditer(mod_rule_str)
        pred_evaluators = get_all_predicate_evaluators()
        for m in pred_matches:
            pred_basename = m.group('pred_name')
            agent_string = m.group('agents')
            agents = re.split(r"_a", agent_string)
            predicate_agent_placeholders = []
            for a in agents:
                if a != '':
                    predicate_agent_placeholders.append(int(a))
            # TODO: Error message
            evaluator = pred_evaluators[pred_basename]
            if m.group('io_type') is None:
                io_type = IOType.OUTPUT
            else:
                io_type = IOType.INPUT
            assert evaluator is not None
            p = PredicateNode(
                    m.group("pred_name") + "_" + m.group("agents"),
                    predicate_agent_placeholders,
                    evaluator(config["traffic_rules_param"]), io_type)
            mod_rule_str = mod_rule_str.replace(m.group(0), p.name)
            predicate_assignment.add(p)
        node = RuleNode(sub_rules + list(predicate_assignment), mod_rule_str, name)
    return node


class Rule:
    full_predicate_pattern = re.compile(r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)?_(?P<agents>(_a(\d)+)+)))")
    quantification_pattern = re.compile(r"^(?P<quant>[AE])\sa(?P<veh_id>\d+):\s\((?P<rule>.*)\)$")
    subrule_pattern = re.compile(r"[AE]\sa\d+:\s\(.*\)")

    class PredicateAssignment:
        def __init__(self, full_name, agent_placeholders, evaluator,
                     io_type=IOType.OUTPUT):
            assert len(
                agent_placeholders) == evaluator.arity, f"The arity of the evaluator for {full_name} should be {len(agent_placeholders)}, but is {evaluator.arity}!"
            self.full_name = full_name
            self.agent_placeholders = tuple(agent_placeholders)
            self.evaluator = evaluator
            self.io_type = io_type

        @property
        def base_name(self):
            return self.evaluator.predicate_name

        @property
        def num_dependencies(self):
            return len(self.agent_placeholders)

        def __eq__(self, o) -> bool:
            return self.full_name == o.full_name and self.agent_placeholders == o.agent_placeholders

        def __hash__(self) -> int:
            return hash((self.full_name, self.agent_placeholders))

    def __init__(self, rule_str, num_dependent_vehicles, quantification, sub_rules, predicate_assignment, config, quantified_vehicle=None, name=None):
        self._rule_str = rule_str
        self.config = config
        self.predicate_assignment = predicate_assignment
        self.num_dependent_vehicles = num_dependent_vehicles
        self.quantification = quantification
        self.name = name
        self.sub_rules = sub_rules
        self.quantified_vehicle = quantified_vehicle

    @property
    def is_vehicle_dependent(self):
        return self.num_dependent_vehicles > 0

    @property
    def predicate_names(self):
        return sorted([pred.full_name for pred in self.predicate_assignment])

    @classmethod
    def from_string(cls, full_rule_str, config, name=None):
        sub_rule_str = []
        mod_rule_str = full_rule_str
        predicate_assignment = set()
        agent_placeholders = set()
        sub_rules = []
        quantification = QuantificationType.NONE
        quantified_vehicle = None
        m = Rule.quantification_pattern.match(mod_rule_str)
        if m is not None:
            # Quantification on top level
            # mod_rule_str = mod_rule_str[
            #                :m.start()] + f"g{len(sub_rule_str)}" + mod_rule_str[
            #                                                        m.end():]
            sub_rule_str = m["rule"]
            sub_rules.append(cls.from_string(sub_rule_str, config, f"g{len(sub_rules)}"))
            if m.group("quant") == "E":
                quantification = QuantificationType.EXISTENTIAL
            elif m.group("quant") == "A":
                quantification = QuantificationType.ALL
            else:
                raise ValueError()
            quantified_vehicle = int(m.group("veh_id"))
        else:
            m = Rule.subrule_pattern.match(mod_rule_str)
            while m is not None:
                mod_rule_str = mod_rule_str[:m.start()] + f"g{len(sub_rule_str)}" + mod_rule_str[m.end():]
                sub_rule_str = m[0]
                sub_rules.append(cls.from_string(sub_rule_str, config, f"g{len(sub_rules)}"))
                m = Rule.subrule_pattern.match(mod_rule_str)


            pred_matches = Rule.full_predicate_pattern.finditer(mod_rule_str)
            pred_evaluators = get_all_predicate_evaluators()
            for m in pred_matches:
                pred_basename = m.group('pred_name')
                agent_string = m.group('agents')
                agents = re.split(r"_a", agent_string)
                predicate_agent_placeholders = []
                for a in agents:
                    if a != '':
                        predicate_agent_placeholders.append(int(a))
                        agent_placeholders.add(int(a))
                # TODO: Error message
                evaluator = pred_evaluators[pred_basename]
                if m.group('io_type') is None:
                    io_type = IOType.OUTPUT
                else:
                    io_type = IOType.INPUT
                assert evaluator is not None
                p = Rule.PredicateAssignment(
                    m.group("pred_name") + "_" + m.group("agents"),
                    predicate_agent_placeholders, evaluator(config["traffic_rules_param"]), io_type)
                mod_rule_str = mod_rule_str.replace(m.group(0), p.full_name)
                predicate_assignment.add(p)

        # Check increasing order
        a_ids = sorted(list(agent_placeholders))
        for i, aid in enumerate(a_ids):
            assert i == aid, f"Agent place holder IDs are not in increasing order. Missing {i}!"
        # List is sorted. Last item is largest.
        num_dependent_vehicles = len(a_ids)
        if name is None:
            name = full_rule_str
        rule = cls(mod_rule_str, num_dependent_vehicles, quantification, sub_rules, predicate_assignment, config, quantified_vehicle, name)
        return rule

class RuleNode:
    def __init__(self, children, rule_str, name):
        self.children = children
        self.name = name
        self.rule_str = rule_str

    def visit(self, visitor):
        return visitor.visit_rule_node(self)

class AllNode:
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle

    def visit(self, visitor):
        return visitor.visit_all_node(self)

class ExistNode:
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle

    def visit(self, visitor):
        return visitor.visit_exist_node(self)

class PredicateNode:
    def __init__(self, full_name, agent_placeholders, evaluator,
                 io_type=IOType.OUTPUT):
        assert len(
            agent_placeholders) == evaluator.arity, f"The arity of the evaluator for {full_name} should be {len(agent_placeholders)}, but is {evaluator.arity}!"
        self.name = full_name
        self.agent_placeholders = tuple(agent_placeholders)
        self.evaluator = evaluator
        self.io_type = io_type

    def visit(self, visitor, *args):
        return visitor.visit_predicate_node(self, *args)

    @property
    def base_name(self):
        return self.evaluator.predicate_name

    @property
    def num_dependencies(self):
        return len(self.agent_placeholders)

    def __eq__(self, o) -> bool:
        return self.name == o.name and self.agent_placeholders == o.agent_placeholders

    def __hash__(self) -> int:
        return hash((self.name, self.agent_placeholders))

    def copy(self):
        return self


# class RuleParserVisitor(StlParserVisitor):
#
#     def __init__(self):
#         self.evaluators = get_all_predicate_evaluators()
#         self.evaluator_config = YAML().load(Path(__file__).parent.parent / "traffic_rules_rtamt.yaml")["traffic_rules_param"]
#
#
#     def defaultResult(self):
#         return []
#
#     def aggregateResult(self, aggregate, nextResult):
#         aggregate.append(nextResult)
#         return aggregate
#
#     def visitExprQuantExist(self, ctx: StlParser.ExprQuantExistContext):
#         children = self.visit(ctx.expression())
#         node = ExistNode(children)
#         return super().visitExprQuantExist(ctx)
#
#     def visitExprQuantForall(self, ctx: StlParser.ExprQuantForallContext):
#         return super().visitExprQuantForall(ctx)
#
#     def visitExprUnquant(self, ctx: StlParser.ExprUnquantContext):
#         return super().visitExprUnquant(ctx)
#
#     def visitPredicate(self, ctx: StlParser.PredicateContext):
#         basename = ctx.Identifier().getText()
#         vehicles = [self.visit(v) for v in ctx.vehicle()]
#         evaluator = self.evaluators[basename](self.evaluator_config)
#         pred = PredicateNode(ctx.getText(), vehicles, evaluator)
#         return pred
#
#     def visitVehicle(self, ctx: StlParser.VehicleContext):
#         veh_id = int(ctx.IntegerLiteral().getText())
#         return veh_id




