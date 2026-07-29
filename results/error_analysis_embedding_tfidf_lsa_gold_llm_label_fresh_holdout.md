# Error Analysis — embedding_tfidf_lsa_gold_llm_label_fresh_holdout

261 false positives and 58 false negatives out of 772 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `neutral_allergen_mention` | 162 | 62% | 169 |
| `illness_mentioned_not_caused_here` | 36 | 14% | 207 |
| `unexplained_fp` | 29 | 11% | 265 |
| `negated_hazard` | 13 | 5% | 226 |
| `generic_complaint_no_hazard` | 11 | 4% | 184 |
| `secondhand_or_hearsay` | 6 | 2% | 278 |
| `hyperbole_or_slang` | 2 | 1% | 297 |
| `strong_negative_sentiment_only` | 1 | 0% | 60 |
| `unpleasant_not_unsafe` | 1 | 0% | 233 |

### `neutral_allergen_mention` — 162 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- First off, let me say that I'm neither vegan nor gluten-free. I like to try vegan foods, though, because when there's a reasonable vegan option, I'll generally go for that. Unfortunately, places like Sweet Freedom are the reason people think vegan baking can't be good. I've baked… (p=0.543)
- I love the food here, it makes for a different gluten free meal in unique surroundings. My only issue is the inconsistent opening hours, they frequently close early, and some weeks don't open at all. The owner/chef is very rude, has zero customer service ability, or maybe just di… (p=0.935)
- It doesn't look like much, but this local place was great for take out. Due to COVID-19, we haven't been going out much, but got real tired of cooking after a whole month. We wanted to support a local business, and were able to have a no contact pick up. Win win all around. I ord… (p=0.392)

### `illness_mentioned_not_caused_here` — 36 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- Ignorant and inconsiderate!!! Do not even attempt to eat here if you have any type of food allergy or dietary restriction. They are completely inconsiderate of making any accommodations. my friend took her daughter here who is lactose intolerant. She brought her own almond milk f… (p=0.965)
- I was traveling on business and I'm giving Tavern an extra star for going above and beyond on accommodating my Alpha-gal food allergy. Service was absolutely outstanding. Unfortunately, the food missed the mark. I had the salmon with which was crusted over and very hard with a ve… (p=0.663)
- Wish I could give them 0 stars because that's what they've earned. Never met such an immature and careless owner. Was greeted 30+ minutes after sitting, took 1+ hour for drinks. Served chips, salsa and water for the table myself. Allergen information incorrect and unclear. Server… (p=0.968)

### `unexplained_fp` — 29 cases

No rule matched — requires manual review.

- Went for the first time for lunch, paid $8 for the buffet. Clean large and open environment. Better than alot of the cramped dark Chinese buffets that are much more common and usually end in food poisoning. The food quality was above average for a buffet and everything I saw at t… (p=0.978)
- Cannot believe this hype monster breathes fire with five star reviews. Maybe if you love salty feces slithering down your throat and surly employees who have to "make nice" when the management is present. Christ. Typical cafeteria routine. Think Lubys on a smaller scale. They're … (p=0.380)
- Finding a good restaurant with vegan options in a new town can be difficult. Siam Elephant was intriguing and within walking distance from our hotel, so we decided to give it a try. Siam Elephant is a Thai restaurant, which raised some concerns for me as someone with severe peanu… (p=0.564)

### `negated_hazard` — 13 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Food is over priced for what you get and server gave me the wrong sandwich! Good thing I'm not allergic to the turkey I didn't want The bread was toasted so much my nePhew cut his mouth! Wait in line and for food is too long! Not worth it at all!!!!!!… (p=0.321)
- One of my go-to's for classic Italian food done right! Some of my favorite dishes include the calamari Sicilian, eggplant tower, grilled romaine salad, eggplant or chicken parm, penne vodka, pumpkin ravioli, sweet potato gnocchis. Their crab cake is my favorite -- served over sau… (p=0.546)
- After reading reviews that Green Basil happily accommodates vegan and vegetarian diets, I ordered delivery specifying that everything must be vegan like this: (***** VEGAN PLEASE *****). Normally if a restaurant cannot accommodate this, they have the courtesy of contacting me to … (p=0.712)

### `generic_complaint_no_hazard` — 11 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- 3.5/5 stars. I really want this place to do well. The man who owns this place is incredibly friendly and makes all the ice cream at the shop. We went during the summer where you would expect ice cream to be relatively higher in demand, but because business was slow, this shop was… (p=0.694)
- When you're on a time crunch, the last thing you need is a slow moving line in a fast food drive-through. As we inched through this particular Steak 'n Shake, we all immediately regretted our decision on choosing this place. Due to being squeezed between two other vehicles, we we… (p=0.371)
- On two separate occasions I asked for no dairy, and twice they got it wrong. It's especially irritating since she repeated it back to me correctly and it's correct on my receipt, and then I bite into it and need to throw it away because I'm lactose intolerant. So don't expect the… (p=0.785)

### `secondhand_or_hearsay` — 6 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- Bar Louie protects customers from unprotected servers by ignoring older patrons Due to Covid, I chose mid-afternoon for a RARE and relished venture out. Arrival time, 3:11 ish. Outdoors, one other table occupied. Two servers. Awesome. A third server appeared after about eight min… (p=0.956)
- A full celebration tea for 2 contains: 2 pots of tea, a fruit plate,salad and soup, 2 tier tea stand and cupcake with icecream as dessert. The environment is very well decorated , and apparently it's someone's home. When I made a reservation they told me there's only 12 and 1 o'c… (p=0.916)
- I'm bordering on 3.5 stars. For some reason I thought this would be a little less fratty than it was. I really enjoyed the booze milkshake (I got the rice crispy one). It's made with graeters ice cream so throw sheets to the wind and abandon any dairy allergy you may have. It was… (p=0.781)

### `hyperbole_or_slang` — 2 cases

Figurative language reusing hazard vocabulary ('to die for', 'killer tacos'). Purely lexical signal with inverted sentiment — the clearest case for contextual embeddings over TF-IDF.

- "Chicken dum biryani? Can I get some smart biryani?" Vomit-inducing, cringe-inducing, jokes aside, that was one of the highlights of my visit to the lunch buffet at Hyderabad House. Today they had several kinds of Biryani (I wish I could remember the names of the others). The kin… (p=0.343)
- Is there any good way to wait for your food at Remedy? They just shout it out when the order is ready, but you don't know if it's actually your food or someone else's who for the same thing. Plus if you're sitting upstairs it's almost impossible to hear anything that's going on d… (p=0.313)

### `strong_negative_sentiment_only` — 1 cases

Highly negative review with no hazard content. The model is partly reading sentiment as hazard, unsurprising given the label used a star-rating gate.

- If I could give this 0 stars it would be more appropriate! Having Celiac is always a struggle to eat out, but I've always managed to find something to eat at every restaurant. Today I was told I could not eat anything in this restaurant!! Really??? Are you just lazy??? I should b… (p=0.820)

### `unpleasant_not_unsafe` — 1 cases

Describes filth or disgust without an adverse event. The boundary is a genuine definitional question — arguably these deserve flagging in a real deployment.

- The Hubby and I did a Delmar Loop date night last night - dinner, movie and desserts. We started the evening at Nico for dinner. I was very curious about this place after reading the reviews on Yelp...and I gotta say, I just don't get it. From a location/ atmosphere/ people watch… (p=0.981)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 36 | 62% | 129 |
| `unsafe_handling_no_illness` | 7 | 12% | 171 |
| `positive_review_with_hazard` | 5 | 9% | 184 |
| `negation_misread` | 4 | 7% | 189 |
| `buried_in_long_review` | 3 | 5% | 182 |
| `contamination_no_illness` | 1 | 2% | 172 |
| `mild_understated_wording` | 1 | 2% | 459 |
| `too_short_weak_signal` | 1 | 2% | 25 |

### `unexplained_fn` — 36 cases

No rule matched — requires manual review.

- Years ago, I frequented this donut shop on my way home from work in the mornings. During this time, they began putting bacon on their maple bars and soon thereafter all their donuts tasted like bacon...this is because they fry the donuts in the SAME VAT OF GREASE in which they fr… (p=0.001)
- I used to love coming here but I've been here four times over the past month and every time my order is wrong. I do not know what happed within a month but they have gone down hill. I have food allergies so it is very important that my order is made correctly yet Karma has failed… (p=0.018)
- I was really excited when this place opened! The hours made it impossible for me to get a cupcake from here. When I finally found a day where I could go, I was disappointed. My expectations were very high and the cupcakes just didn't quite meet them. The cupcakes were very dense … (p=0.005)

### `unsafe_handling_no_illness` — 7 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- I first ordered from here last week and it was super delicious and I was very happy with my order. However, tonight I ordered a salad and I thought I tasted meat in a couple of bites but I didn't see any until I was almost finished eating. I saw what looked like little bits of ba… (p=0.028)
- I have been to this restaurant twice now. The first time was about as perfect as you can imagine. The food was perfect the prime rib perfectly cooked and seasoned, the waitstaff helpful and polite. Literally everything was perfect. It was probably the best prime rib I have ever h… (p=0.003)
- Often times I have come here over the years as they have a great selection of organic and gluten free options. I love the wide array of pre-prepared foods/freezer items and even micro brew beer selections. The staff has always been wonderful and I have had no problem checking out… (p=0.034)

### `positive_review_with_hazard` — 5 cases

4-5 star review reporting a hazard. The star gate in the labelling rule means the training data barely contains these, so the model associates hazards with low ratings.

- ***This is more of a 3.5 place but I rounded up*** Okay, so I was in Nashville last weekend for a friend's bachelorette party. For our first night in town, we came here for dinner. I'll start by saying that the building itself is super swanky. It looked small from the outside but… (p=0.059)
- Really impressed at El Sur today. Inconspicuous, colorful building - definitely one of those Don't judge a book by its cover scenarios. Very friendly staff and always smiling. I ordered the fish tacos and my husband ordered the shredded beef tacos. The waitress warned me about th… (p=0.003)
- Your review helps others learn about great local businesses. Was in the house the last time the Eagles won the NFC championship and it was raucous. loved the crab fries, would only be better if they were house cute. Also had the crab and it was above average. Got there around 1pm… (p=0.000)

### `negation_misread` — 4 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- The icing tasted great, the cake however was a little too dry - this day, the cupcakes were either over baked or too old or not stored properly? I bought a dozen for my office for around $32. We all had the same complaint on the cake being too dry in this batch. I tried this plac… (p=0.045)
- You know, it's really too bad that I have to give the Hookah House a three-start review. The place is beautiful, and I've often said that it seems like a place that James Bond would have visited. If Bond were to have visited the Hookah House, he'd have enjoyed a seat in one of th… (p=0.014)
- Server brought cocktails and there was paper towel pieces in them... We told the server... He took them away and came back with the same drinks. Bartender said it was the oil from the olives. NO there is Positively PAPER in my cocktail. Finally got them to make them over but wait… (p=0.183)

### `buried_in_long_review` — 3 cases

Hazard mentioned late in a long review, or diluted by surrounding content. NOTE: the tempting 'past the 256-token window' reading was MEASURED AND REFUTED for the gold false negatives — with the real DebertaV2TokenizerFast at max_length=256, only 1 of 23 residual FNs has its hazard cue past the window, only 2 of 23 exceed 256 tokens at all, and the median cue position is token 39. Raising max_length would recover at most one FN. See results/gold_fn_handread.md. Treat this bucket as dilution/salience, not truncation.

- i don't think i could live without their breakfast burrito (the barking dog) It is so good. I'm not a big fan of sweet stuff in the morning but their french toast with bananas or pancakes with fruit stuff OR even whatever they are having as a special is always good. I have also l… (p=0.003)
- My Husband and I visit PF Chang yesterday for dinner. My husband ordered the Kung Pao chicken with ice tea and I ordered the pepper steak, for appetizer we ordered the crispy green beans. The crispy green beans was amazing and the drinks (coconut cooler) was really good. My husba… (p=0.012)
- Disappointing, un-imaginative, bland food. Especially from a chef with another well-loved restaurant in town. From the blog upon opening in 2014, Owner Woolsey writes "To me, Bistrot La Minette and La Peg are members of my family. They're babies that require nurturing and guidanc… (p=0.004)

### `contamination_no_illness` — 1 cases

A foreign object or tampering, without illness vocabulary. Same inherited blind spot as above; this is where the heuristic's recall is weakest (88.5% on contamination).

- Twice! Two times too many I bought a large bag of new dry dog food at this store, only to have insects crawling around the kibble and the moths flying out and latching on to the walls of my house. It was gross. I've bought bags at the grocery store and other Pet stores, and this … (p=0.183)

### `mild_understated_wording` — 1 cases

Understated symptoms ('didn't sit right with me'). Same ceiling as above, milder.

- I've been wanting to check this place out for for brunch for months now. I finally did and man am I disappointed. The dining experience got worse and worse with every passing moment. First-the $12 Bloody Mary "bar." For $12 I thought there would be more to the make your own blood… (p=0.077)

### `too_short_weak_signal` — 1 cases

Very short review; little evidence either way.

- Started a new job in the area and wanted some Chinese. Picked up some lunch from here and Almost threw up afterwards. Will not return!… (p=0.001)
