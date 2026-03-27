"""
Event Rules Module.

Rule-based event filtering and priority management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

from .detector import DetectedEvent, EventType, EventSignal


class EventPriority(Enum):
    """Priority levels for events."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class EventRule:
    """Defines a rule for event handling."""
    name: str
    event_types: List[EventType]
    priority: EventPriority
    min_severity: float = 0.0
    max_age_seconds: float = 3600  # Ignore events older than this
    cooldown_seconds: float = 300  # Minimum time between similar events
    action: str = "process"  # process, ignore, alert
    conditions: Dict = field(default_factory=dict)


class EventRuleEngine:
    """
    Rule engine for filtering and prioritizing events.
    
    Applies rules to:
    - Filter out low-priority events
    - Set event priorities
    - Enforce cooldowns
    - Trigger actions
    """
    
    def __init__(self):
        self._last_event_times: Dict[str, datetime] = {}
        self._rules = self._get_default_rules()
    
    def _get_default_rules(self) -> List[EventRule]:
        """Get default event rules."""
        return [
            EventRule(
                name="high_severity_congestion",
                event_types=[EventType.CONGESTION_SPIKE],
                priority=EventPriority.HIGH,
                min_severity=0.7,
                cooldown_seconds=600,
                action="process"
            ),
            EventRule(
                name="major_port_delay",
                event_types=[EventType.PORT_DELAY],
                priority=EventPriority.MEDIUM,
                min_severity=0.5,
                cooldown_seconds=1800,
                action="process"
            ),
            EventRule(
                name="significant_breakout",
                event_types=[EventType.PRICE_BREAKOUT],
                priority=EventPriority.HIGH,
                min_severity=0.6,
                cooldown_seconds=300,
                action="process"
            ),
            EventRule(
                name="volume_confirmation",
                event_types=[EventType.VOLUME_SPIKE],
                priority=EventPriority.MEDIUM,
                min_severity=0.5,
                cooldown_seconds=600,
                action="process"
            ),
            EventRule(
                name="negative_news_filter",
                event_types=[EventType.NEGATIVE_NEWS],
                priority=EventPriority.MEDIUM,
                min_severity=0.4,
                cooldown_seconds=1800,
                action="process"
            ),
            EventRule(
                name="positive_news_filter",
                event_types=[EventType.POSITIVE_NEWS],
                priority=EventPriority.MEDIUM,
                min_severity=0.4,
                cooldown_seconds=1800,
                action="process"
            ),
            EventRule(
                name="correlation_alert",
                event_types=[EventType.CORRELATION_BREAK],
                priority=EventPriority.HIGH,
                min_severity=0.5,
                cooldown_seconds=3600,
                action="alert"
            ),
            EventRule(
                name="route_deviation",
                event_types=[EventType.ROUTE_DEVIATION],
                priority=EventPriority.LOW,
                min_severity=0.3,
                cooldown_seconds=3600,
                action="process"
            ),
        ]
    
    def apply_rules(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Apply rules to filter and process events."""
        processed_events = []
        
        for event in events:
            result = self._apply_event_rules(event)
            if result.should_process:
                processed_events.append(event)
        
        return processed_events
    
    def _apply_event_rules(self, event: DetectedEvent) -> "RuleResult":
        """Apply rules to a single event."""
        applicable_rules = [
            rule for rule in self._rules
            if event.event_type in rule.event_types
        ]
        
        if not applicable_rules:
            # No rule matched - use default processing
            return RuleResult(should_process=True, priority=EventPriority.LOW)
        
        # Use highest priority matching rule
        best_rule = min(applicable_rules, key=lambda r: r.priority.value)
        
        # Check severity threshold
        if event.severity < best_rule.min_severity:
            return RuleResult(should_process=False, priority=best_rule.priority)
        
        # Check cooldown
        cooldown_key = f"{best_rule.name}:{event.event_type.value}"
        if cooldown_key in self._last_event_times:
            last_time = self._last_event_times[cooldown_key]
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < best_rule.cooldown_seconds:
                return RuleResult(should_process=False, priority=best_rule.priority)
        
        # Check age
        age = (datetime.now() - event.timestamp).total_seconds()
        if age > best_rule.max_age_seconds:
            return RuleResult(should_process=False, priority=best_rule.priority)
        
        # Update last event time
        self._last_event_times[cooldown_key] = datetime.now()
        
        return RuleResult(
            should_process=best_rule.action == "process",
            priority=best_rule.priority,
            action=best_rule.action
        )
    
    def add_rule(self, rule: EventRule):
        """Add a new rule."""
        self._rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self._rules = [r for r in self._rules if r.name != name]
    
    def get_rules_for_event(self, event_type: EventType) -> List[EventRule]:
        """Get all rules applicable to an event type."""
        return [r for r in self._rules if event_type in r.event_types]
    
    def prioritize_events(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Sort events by priority."""
        def get_priority_key(event):
            rule_result = self._apply_event_rules(event)
            return (0 if rule_result.should_process else 1, rule_result.priority.value)
        
        return sorted(events, key=get_priority_key)
    
    def reset_cooldowns(self):
        """Reset all cooldown timers."""
        self._last_event_times.clear()


@dataclass
class RuleResult:
    """Result of applying rules to an event."""
    should_process: bool
    priority: EventPriority
    action: str = "process"


@dataclass
class EventAggregator:
    """Aggregates similar events over a time window."""
    
    window_seconds: float = 300  # 5 minute window
    min_events: int = 2
    
    def aggregate(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Aggregate events within the time window."""
        if not events:
            return []
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        aggregated = []
        current_group = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            time_diff = (event.timestamp - current_group[0].timestamp).total_seconds()
            
            if time_diff <= self.window_seconds:
                current_group.append(event)
            else:
                # Process current group
                if len(current_group) >= self.min_events:
                    aggregated.append(self._combine_events(current_group))
                current_group = [event]
        
        # Process last group
        if len(current_group) >= self.min_events:
            aggregated.append(self._combine_events(current_group))
        
        return aggregated
    
    def _combine_events(self, events: List[DetectedEvent]) -> DetectedEvent:
        """Combine multiple events into one."""
        base = events[0]
        
        # Use highest severity
        max_severity = max(e.severity for e in events)
        
        # Combine descriptions
        descriptions = [e.description for e in events[:3]]
        combined_desc = f"{len(events)} events: " + "; ".join(descriptions)
        
        return DetectedEvent(
            event_id=f"AGG-{base.event_id}",
            event_type=base.event_type,
            timestamp=base.timestamp,
            severity=max_severity,
            description=combined_desc,
            raw_data={"aggregated_count": len(events), "events": [e.event_id for e in events]}
        )


# Singleton
_rule_engine = None

def get_rule_engine() -> EventRuleEngine:
    """Get or create the rule engine singleton."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = EventRuleEngine()
    return _rule_engine
