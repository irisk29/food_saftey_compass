# Error Analysis — embedding_tfidf_lr_gold_llm_label_fresh_holdout

162 false positives and 68 false negatives out of 772 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `neutral_allergen_mention` | 73 | 45% | 176 |
| `illness_mentioned_not_caused_here` | 39 | 24% | 185 |
| `unexplained_fp` | 21 | 13% | 252 |
| `generic_complaint_no_hazard` | 12 | 7% | 168 |
| `negated_hazard` | 12 | 7% | 207 |
| `secondhand_or_hearsay` | 2 | 1% | 308 |
| `hyperbole_or_slang` | 1 | 1% | 181 |
| `strong_negative_sentiment_only` | 1 | 1% | 60 |
| `unpleasant_not_unsafe` | 1 | 1% | 233 |

### `neutral_allergen_mention` — 73 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- Hoe excited was i to find a gluten free vegan pizza (up charge) option. Staff was super accommodating and courteous. When ordering gluten free it is common to be asked Allergy or Preferance. But when i get down the line to add my toppings they actually switched gloves to prevent … (p=1.000)
- Great food! The burger and fries were some of the best I've ever had. If I eat beef these days, I try to restrict it to grass fed, so was delighted to find this gem while on the road. Wonderful slow fast food. My travel companion is sensitive to gluten and always has to order bur… (p=0.999)
- I'm gluten intolerant so this was on my list of restaurants in Nola that provide a gluten free menu. Most of the regular dinner menu with a few tweaks here and there I was able to have. I chose the fresh oysters to start which were very good although a couple hard pretty big chun… (p=0.883)

### `illness_mentioned_not_caused_here` — 39 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- Ignorant and inconsiderate!!! Do not even attempt to eat here if you have any type of food allergy or dietary restriction. They are completely inconsiderate of making any accommodations. my friend took her daughter here who is lactose intolerant. She brought her own almond milk f… (p=0.996)
- Ice cream was awsome but more importantly my daughter has severe food allergies. Staff were knowledgeable and had a good process preventing cross contamination.… (p=1.000)
- I was traveling on business and I'm giving Tavern an extra star for going above and beyond on accommodating my Alpha-gal food allergy. Service was absolutely outstanding. Unfortunately, the food missed the mark. I had the salmon with which was crusted over and very hard with a ve… (p=0.421)

### `unexplained_fp` — 21 cases

No rule matched — requires manual review.

- Went for the first time for lunch, paid $8 for the buffet. Clean large and open environment. Better than alot of the cramped dark Chinese buffets that are much more common and usually end in food poisoning. The food quality was above average for a buffet and everything I saw at t… (p=0.517)
- Absolute waste of time. Poorly managed. Was doing our normal bounce around & eat apps at restaurants we haven't been to in a while & stepped in for a drink at midnight. . We wanted to sit in one of the 6 vacant hi-top booths in the bar area. The young hostess told us it was only … (p=0.790)
- This little brewery ain't so bad! This brewery is located in a pretty industrial area and seems rather huge. My husband and I sat at the bar and was greeted by a very friendly bartender. I ordered the Tucson Blonde which was nothing to write home about. In fact, it was a little f… (p=0.430)

### `generic_complaint_no_hazard` — 12 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- 3.5/5 stars. I really want this place to do well. The man who owns this place is incredibly friendly and makes all the ice cream at the shop. We went during the summer where you would expect ice cream to be relatively higher in demand, but because business was slow, this shop was… (p=0.366)
- Nice clean store with great selection of organics. People at the register are always friendly too and love that their bakery super careful about cross contamination when it comes to peanuts...at least so they say when I expressed the concern.… (p=1.000)
- When it comes to fresh sushi and authentic Asian cuisine dining in the suburbs. I definitely recommend this place. Staff are very friendly and accommodating to our requests. Their special rolls are delicious! With one bite of the lobster roll, my mouth was bursting with exciting … (p=0.996)

### `negated_hazard` — 12 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Food is over priced for what you get and server gave me the wrong sandwich! Good thing I'm not allergic to the turkey I didn't want The bread was toasted so much my nePhew cut his mouth! Wait in line and for food is too long! Not worth it at all!!!!!!… (p=0.519)
- One of my go-to's for classic Italian food done right! Some of my favorite dishes include the calamari Sicilian, eggplant tower, grilled romaine salad, eggplant or chicken parm, penne vodka, pumpkin ravioli, sweet potato gnocchis. Their crab cake is my favorite -- served over sau… (p=1.000)
- After reading reviews that Green Basil happily accommodates vegan and vegetarian diets, I ordered delivery specifying that everything must be vegan like this: (***** VEGAN PLEASE *****). Normally if a restaurant cannot accommodate this, they have the courtesy of contacting me to … (p=0.999)

### `secondhand_or_hearsay` — 2 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- I'm bordering on 3.5 stars. For some reason I thought this would be a little less fratty than it was. I really enjoyed the booze milkshake (I got the rice crispy one). It's made with graeters ice cream so throw sheets to the wind and abandon any dairy allergy you may have. It was… (p=0.985)
- The restaurant was inconsistent at best. I went for the lunch buffet. The food was actually pretty good. The food was tasty and fish was fresh. The sashimi was actually normal thickness. There were also a hot food selection. The problem was there isn't that many people that go fo… (p=1.000)

### `hyperbole_or_slang` — 1 cases

Figurative language reusing hazard vocabulary ('to die for', 'killer tacos'). Purely lexical signal with inverted sentiment — the clearest case for contextual embeddings over TF-IDF.

- We had an amazing dinner last Friday, in part and thanks to our amazing server in the bar, Brittney. Can't say we expected much despite the 20minute wait and thought it would be like every other "chain" but it was anything but. I have a severe Paprika allergy and eating out can b… (p=0.968)

### `strong_negative_sentiment_only` — 1 cases

Highly negative review with no hazard content. The model is partly reading sentiment as hazard, unsurprising given the label used a star-rating gate.

- If I could give this 0 stars it would be more appropriate! Having Celiac is always a struggle to eat out, but I've always managed to find something to eat at every restaurant. Today I was told I could not eat anything in this restaurant!! Really??? Are you just lazy??? I should b… (p=1.000)

### `unpleasant_not_unsafe` — 1 cases

Describes filth or disgust without an adverse event. The boundary is a genuine definitional question — arguably these deserve flagging in a real deployment.

- The Hubby and I did a Delmar Loop date night last night - dinner, movie and desserts. We started the evening at Nico for dinner. I was very curious about this place after reading the reviews on Yelp...and I gotta say, I just don't get it. From a location/ atmosphere/ people watch… (p=0.997)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 43 | 63% | 160 |
| `unsafe_handling_no_illness` | 10 | 15% | 213 |
| `buried_in_long_review` | 6 | 9% | 222 |
| `negation_misread` | 4 | 6% | 130 |
| `positive_review_with_hazard` | 3 | 4% | 135 |
| `contamination_no_illness` | 1 | 2% | 172 |
| `too_short_weak_signal` | 1 | 2% | 25 |

### `unexplained_fn` — 43 cases

No rule matched — requires manual review.

- Years ago, I frequented this donut shop on my way home from work in the mornings. During this time, they began putting bacon on their maple bars and soon thereafter all their donuts tasted like bacon...this is because they fry the donuts in the SAME VAT OF GREASE in which they fr… (p=0.000)
- Service was really bad. Salad ordered without sprouts. It came out with sprouts on it. Sent it back. The waiter brought the same salad back from the kitchen after trying to pick all the sprouts off. Which is both gross and dangerous because of food allergies. The noise level was … (p=0.015)
- Chase appears nice on the outside, but the food was mediocre at best, certainly not worth the price. My mom got terrible food poisoning from the ravioli that kept her up all night. The vodka pappardelle and eggplant parm were good, but swimming in sauce. If you are considering It… (p=0.180)

### `unsafe_handling_no_illness` — 10 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- I was not impressed. Prices are high and why have a lobster tank if it's empty? Too many gimmick rolls on the menu. If there are no live lobsters in the lobster tank, where are they getting lobsters for their fancy raw lobster roll? I'm guessing way dead frozen ones in the back. … (p=0.015)
- Often times I have come here over the years as they have a great selection of organic and gluten free options. I love the wide array of pre-prepared foods/freezer items and even micro brew beer selections. The staff has always been wonderful and I have had no problem checking out… (p=0.018)
- I'm giving it two stars just because their staff was awesome and the bloody mary's were good. Other than that, their bacon has literally made my husband and I, never want bacon again. Do NOT order the Bacon Sampler. Like, how do you mess up bacon? All of the flavors tasted like r… (p=0.001)

### `buried_in_long_review` — 6 cases

Hazard mentioned late in a long review, or diluted by surrounding content. NOTE: the tempting 'past the 256-token window' reading was MEASURED AND REFUTED for the gold false negatives — with the real DebertaV2TokenizerFast at max_length=256, only 1 of 23 residual FNs has its hazard cue past the window, only 2 of 23 exceed 256 tokens at all, and the median cue position is token 39. Raising max_length would recover at most one FN. See results/gold_fn_handread.md. Treat this bucket as dilution/salience, not truncation.

- i don't think i could live without their breakfast burrito (the barking dog) It is so good. I'm not a big fan of sweet stuff in the morning but their french toast with bananas or pancakes with fruit stuff OR even whatever they are having as a special is always good. I have also l… (p=0.001)
- My Husband and I visit PF Chang yesterday for dinner. My husband ordered the Kung Pao chicken with ice tea and I ordered the pepper steak, for appetizer we ordered the crispy green beans. The crispy green beans was amazing and the drinks (coconut cooler) was really good. My husba… (p=0.002)
- Came here for breakfast after a night of drinking with my friends following a wedding. Walked in and it seemed everyone knew each other by first name. The customers and workers seemed to all be on a first name basis which was cool. It has that crappy diner feel which I love. If d… (p=0.000)

### `negation_misread` — 4 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- The icing tasted great, the cake however was a little too dry - this day, the cupcakes were either over baked or too old or not stored properly? I bought a dozen for my office for around $32. We all had the same complaint on the cake being too dry in this batch. I tried this plac… (p=0.037)
- I'm shocked at the reviewer who gave this place 5 stars. Were they eating at the same restaurant I was??? The service was absolutely deplorable...unless, of course, you like being insulted and treated rudely while dining out. The food was NOT authentic German food, and 2 people i… (p=0.006)
- Server brought cocktails and there was paper towel pieces in them... We told the server... He took them away and came back with the same drinks. Bartender said it was the oil from the olives. NO there is Positively PAPER in my cocktail. Finally got them to make them over but wait… (p=0.036)

### `positive_review_with_hazard` — 3 cases

4-5 star review reporting a hazard. The star gate in the labelling rule means the training data barely contains these, so the model associates hazards with low ratings.

- Really impressed at El Sur today. Inconspicuous, colorful building - definitely one of those Don't judge a book by its cover scenarios. Very friendly staff and always smiling. I ordered the fish tacos and my husband ordered the shredded beef tacos. The waitress warned me about th… (p=0.000)
- Your review helps others learn about great local businesses. Was in the house the last time the Eagles won the NFC championship and it was raucous. loved the crab fries, would only be better if they were house cute. Also had the crab and it was above average. Got there around 1pm… (p=0.000)
- I read the few reviews and decided I would give it a try... So on Wednesday that just past I decided to go and give it a try mind you I'm a very picky eater don't just eat from every where..was Food Poisoned from eating Spanish Food, so I watch where I eat!! So I ordered and deci… (p=0.017)

### `contamination_no_illness` — 1 cases

A foreign object or tampering, without illness vocabulary. Same inherited blind spot as above; this is where the heuristic's recall is weakest (88.5% on contamination).

- Twice! Two times too many I bought a large bag of new dry dog food at this store, only to have insects crawling around the kibble and the moths flying out and latching on to the walls of my house. It was gross. I've bought bags at the grocery store and other Pet stores, and this … (p=0.002)

### `too_short_weak_signal` — 1 cases

Very short review; little evidence either way.

- Started a new job in the area and wanted some Chinese. Picked up some lunch from here and Almost threw up afterwards. Will not return!… (p=0.010)
