from dataclasses import dataclass
from typing import List

from novelrag.model.outline import Event, Outline


@dataclass
class EventLocation:
    path: List[int]  # Indices to current event [0,1,2] means root[0]->subevents[1]->subevents[2]
    
    def __str__(self) -> str:
        return '.'.join(map(str, self.path))
    
    def parent(self) -> 'EventLocation':
        return EventLocation(self.path[:-1])
        
    def child(self, index: int) -> 'EventLocation':
        return EventLocation(self.path + [index])

    @staticmethod
    def _events_of(event: Outline | Event):
        if isinstance(event, Outline):
            return event.events
        else:
            return event.subEvents


    def get_current_event(self, outline: Outline) -> Outline | Event:
        current_event = outline
        for idx in self.path:
            current_event = self._events_of(current_event)[idx]
        return current_event

    def get_current_events(self, outline: Outline) -> list[Event]:
        current_event = self.get_current_event(outline)
        return self._events_of(current_event)

    def get_path_names(self, outline: Outline) -> list[str]:
        """Get the names of events along the current path."""
        names = []
        current_events = outline.events

        for idx in self.path:
            if idx >= len(current_events):
                raise ValueError(f"Index out of bounds: {idx} for {len(current_events)} events")
            event = current_events[idx]
            names.append(event.name)
            current_events = event.subEvents

        return names
