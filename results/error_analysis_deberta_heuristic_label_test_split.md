# Error Analysis — deberta_heuristic_label_test_split

27 false positives and 16 false negatives out of 1500 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `generic_complaint_no_hazard` | 12 | 44% | 261 |
| `illness_mentioned_not_caused_here` | 6 | 22% | 202 |
| `neutral_allergen_mention` | 3 | 11% | 224 |
| `unexplained_fp` | 3 | 11% | 314 |
| `negated_hazard` | 2 | 7% | 204 |
| `secondhand_or_hearsay` | 1 | 4% | 460 |

### `generic_complaint_no_hazard` — 12 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- This is the worst bar experience I've ever had. I'll try to be brief. The three beers I tried to order last night were Sierra Nevada, Sweetwater 420, and Sam Adams Boston Lager. I drink these beers regularly draft and bottle and I am very familiar with them. I ordered a Sierra Ne… (p=0.974)
- I came here with my family today with total of 5 people. Service was decent and food was okay. When the bill came I noticed there was 18% gratuity already added to the bill. I questioned the server where this automatic gratuity info was posted since it was no where to be found in… (p=0.247)
- Husband and I made a mistake and came here on a Lenten Friday. The wait was forever. But since we were already there, might as well suck it up and see what's it all about. We ordered the blackened alligator and oysters for appetizers, the alligator came out within 15 minutes, and… (p=0.325)

### `illness_mentioned_not_caused_here` — 6 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- First impression: Home Wine Kitchen's "No Menu Monday" is a must-try for anyone interested in the St. Louis fine dining scene or good food more generally. We came prepared with reservations, but the restaurant actually began quieting out while our delicious meal was ramping up; t… (p=0.994)
- Found there to be a good variety of food for a late night dinner with young kids. Wish we had one this good near us in upper PA. Person that rang us in was friendly and helpful. Two things made our experience top notch. First was out waiter Eric, who was friendly, observant, went… (p=0.995)
- Smokey (to be expected - minus a star as a non smoker with allergies), several dining options (prices varies), beverages (all sorts for a nominal fee, of course), hotel (pricey), parking (FREE), entertainment (various), ATM (scared for some people who may not have self control), … (p=0.994)

### `neutral_allergen_mention` — 3 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- Gluten free pasta...I had to try. Plus, 1/2 price wine Mon-Thurs did not disappoint. The selection was just right. We found street parking as the lot was full, and at 7pm waited a bit for a table as they don't accept reservations for parties of less than 6. To start, we ordered t… (p=0.983)
- 5 stars, as far as sports bars go. Awesome atmosphere with tons of TVs, high ceilings which allows for huge projectors to show games, great service, prices and food. For my gluten free eaters, the tortilla chips and fries are both gluten free. They also use a separate fryer from … (p=0.990)
- The old saying is true! You get what you pay for! Not a tea bag place or Starbucks Ernie, manager, and Jennifer are very knowledgeable of teas from all over the world and how to meet strict standards If you have to ask how much, you can't afford it! Go get Lipton tea and be happy… (p=0.821)

### `unexplained_fp` — 3 cases

No rule matched — requires manual review.

- Terrace Lounge appears to want to have a great cocktail venue, but they fall short. One bartender didn't even know there was a new cocktail menu. The cocktail was weak and watery, and the liquors they have here and throughout Peppermill are the most unfortunate corporate types. T… (p=0.985)
- Worst restaurant experience ever. First, this place is super cramped! The noise level was atrocious. I don't know why they haven't done anything about this. We could not engage in an enjoyable conversation. Second, my plate had dried up food on it from the previous guest. I under… (p=0.646)
- In a direct Bobby Flay vs. Jose Garces match, i.e. comparing Bobby's Burger Palace and Village Whiskey, I'd say BBP clearly wins. Village Whiskey's biggest weakness is actually its high price. Taste-wise, my medium-rare burger with goat cheese and egg is comparable to BBP or 500 … (p=0.991)

### `negated_hazard` — 2 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Although there are countless little Mexican places below Washington Avenue, I never get sick of them and always look forward to trying the new kid on the block. This time it was El Sabor Poblano on Federal, which could not look less special if it tried. I liked that there were 3 … (p=0.995)
- Where do I start. I ordered food to go it was my first time here so I ordered the Blue Jawn burger no lie I vomited we had to hurry up and pull over. I ordered the general tso chicken tacos not good and the sauce was too sweet. They were using bare hands to prepare the food which… (p=0.989)

### `secondhand_or_hearsay` — 1 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- Ruined our Friday happiness. Thought it would be nice to go to Friendly's for a family meal to start the weekend...but the moment we arrived, I knew we should have left. Two employees at the front were talking to themselves, ignoring to greet us. When a 3rd employee asked his col… (p=0.285)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 9 | 56% | 425 |
| `buried_in_long_review` | 4 | 25% | 333 |
| `negation_misread` | 3 | 19% | 500 |

### `unexplained_fn` — 9 cases

No rule matched — requires manual review.

- My wife and I went last night for the first time because we are considering moving to Seminole Heights. Before I write a full review I will say the food was great, but the service sub-par. The first thing we noticed as we walked in was a sign in sheet and there were only 4 names … (p=0.006)
- Wow. Where to start? If you don't know, this hotel is in its own zip code - about the size of the community college I attended. Luckily, I was pre-warned of its labyrinthine layout and sprawling floor plan far in advance by a colleague - who mentioned that she never set foot outs… (p=0.038)
- I have actually been here a few times, but since I haven't visited this place in about 3 years, I am going to be basing my review on my most recent experience: yesterday's. So I'm with my big group, a party of 7 to be exact, and we are seated at the first table we laid eyes upon,… (p=0.028)

### `buried_in_long_review` — 4 cases

Hazard mentioned late in a long review, past the 256-token truncation window or diluted by surrounding content. Directly actionable: raise max_length.

- Edley's features a wide variety of barbecue options and sides. It's an attractive restaurant with a lot of seating. Parking was easy later in the evening, but I imagine would be quite difficult during busy times. The portions of meat are generous, the sides less so. Overall, the … (p=0.003)
- I only give one star because I have to!!!!!!!!!! this place is the (excuse my language) biggest piece of shit place I have ever been to!!!!!! Not just the horrible service of the people who even after you order online & it says pick up in 30 minutes & then you get there & somethi… (p=0.033)
- Wonderful atmosphere; live music, cozy seating (and cozy is a nice way of saying that tables are crammed into every nook and cranny in the joint and everyone who walked behind me collided into my chair 30 times) but HEY, that's ok. :) The service was a bit snooty, but not terribl… (p=0.005)

### `negation_misread` — 3 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- Updated 10/4. Had to remove a star because I emailed Morgan (who responded and asked me to contact her) over three weeks ago and haven't heard back. I wasn't looking for a free meal, but I would rather not have taken the time to contact GG if no one planned to respond. Also, I wa… (p=0.158)
- We needed a place to eat a late lunch/early dinner and Luke was recommended by the front desk of the hotel. I read the reviews after and think I still would have given it a try, even though some folks didn't like it. The review that sums it up is a Yelper that said he went here t… (p=0.140)
- I had boycotted this Bully's on Pyramid Hwy for 2-3 years for the bad service, horrible food, complete disrespect for the customers, rude managers, etc. I gave it another try after I heard they finally hired a real chef to plan the menu, upgraded the food quality and hired new ba… (p=0.009)
