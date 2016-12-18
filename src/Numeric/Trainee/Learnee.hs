{-# LANGUAGE RankNTypes, FlexibleContexts, FlexibleInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Numeric.Trainee.Learnee (
	eval,
	(⇉), (⇉′), (′⇉), (⥤), (‖), into, paired,
	cost, squared,
	learneeT, computeeT,
	learnee, computee,

	makeBatches, shuffleList,

	miss, avg,
	runLearnT, runLearn,
	learnPass, trainOnce, trainBatch, trainEpoch, trainUntil
	) where

import Prelude.Unicode

import Control.Arrow ((***))
import Control.Lens
import Control.Monad.State
import Control.Monad.Writer
import Data.List (unfoldr)
import Data.Random (runRVar, StdRandom(..), MonadRandom)
import Data.Random.Internal.Source
import Data.Random.List (shuffle)
import Numeric.AD (AD)
import Numeric.AD.Mode.Forward (Forward, auto, diff')

import Numeric.Trainee.Types

eval ∷ Learnee a b → a → b
eval (Learnee ws f) = fst ∘ f ws

(⇉) ∷ LearneeT w a b → LearneeT w' b c → LearneeT (Chain w w') a c
LearneeT lws f ⇉ LearneeT rws g = LearneeT (Chain (lws, rws)) h where
	h (Chain (lws', rws')) x = (z, up) where
		(y, f') = f lws' x
		(z, g') = g rws' y
		up dz = (dx, Chain (lws'', rws'')) where
			(dy, rws'') = g' dz
			(dx, lws'') = f' dy

(⇉′) ∷ LearneeT w a b → LearneeT NoParams b c → LearneeT w a c
l ⇉′ r = mapP (iso (fst ∘ getChain) (Chain ∘ flip (,) NoParams)) $ l ⇉ r

(′⇉) ∷ LearneeT NoParams a b → LearneeT w b c → LearneeT w a c
l ′⇉ r = mapP (iso (snd ∘ getChain) (Chain ∘ (,) NoParams)) $ l ⇉ r

(⥤) ∷ Learnee a b → Learnee b c → Learnee a c
l ⥤ r = withLearnee l $ \l' → withLearnee r $ \r' → toLearnee (l' ⇉ r')

into ∷ LearneeT w a b → LearneeT w' b c → LearneeT (Chain w w') a c
into = (⇉)

(‖) ∷ LearneeT w a b → LearneeT w' a' b' → LearneeT (Chain w w') (a, a') (b, b')
LearneeT lws f ‖ LearneeT rws g = LearneeT (Chain (lws, rws)) h where
	h (Chain (lws', rws')) (x, y) = ((x', y'), up) where
		(x', f') = f lws' x
		(y', g') = g rws' y
		up (dx', dy') = ((dx, dy), Chain (lws'', rws'')) where
			(dx, lws'') = f' dx'
			(dy, rws'') = g' dy'

paired ∷ LearneeT w a b → LearneeT w' a' b' → LearneeT (Chain w w') (a, a') (b, b')
paired = (‖)

cost ∷ Num a ⇒ (forall s . AD s (Forward a) → AD s (Forward a) → AD s (Forward a)) → Cost a
cost fn y' = diff' (fn (auto y'))

squared ∷ Num a ⇒ Cost a
squared = cost $ \y' y → (y - y') ^ (2 ∷ Integer)

learneeT ∷ Gradee (a, w) b → w → LearneeT w a b
learneeT g ws = LearneeT ws h where
	h ws' x = (y, back) where
		y = view (runGradee g) (x, ws')
		back dy = set (runGradee g) dy (x, ws')

learnee ∷ Parametric w ⇒ Gradee (a, w) b → w → Learnee a b
learnee g ws = toLearnee $ learneeT g ws

computeeT ∷ Gradee a b → LearneeT NoParams a b
computeeT g = LearneeT NoParams h where
	h _ x = (y, back) where
		y = view (runGradee g) x
		back dy = (set (runGradee g) dy x, NoParams)

computee ∷ Gradee a b → Learnee a b
computee g = toLearnee $ computeeT g

makeBatches ∷ Int → [a] → [[a]]
makeBatches sz = takeWhile (not ∘ null) ∘ unfoldr (Just ∘ splitAt sz)

shuffleList ∷ MonadRandom m ⇒ [a] → m [a]
shuffleList ls = runRVar (shuffle ls) StdRandom

miss ∷ Learnee a b → Cost b → Example a b → b
miss l c (x, y') = withLearnee l miss' where
	miss' lt = snd $ learnPass lt c (x, y')

avg ∷ Fractional a ⇒ [a] → a
avg ls = sum ls / fromIntegral (length ls)

runLearnT ∷ Learnee a b → StateT (Learnee a b) m c → m (c, Learnee a b)
runLearnT = flip runStateT

runLearn ∷ Learnee a b → State (Learnee a b) c → (c, Learnee a b)
runLearn = flip runState

learnPass ∷ LearneeT w a b → Cost b → Example a b → (w, b)
learnPass (LearneeT ws f) c (x, y') = (dws, e) where
	(y, back) = f ws x
	(e, de) = c y' y
	(_, dws) = back de

trainOnce ∷ MonadState (Learnee a b) m ⇒ Rational → Cost b → Example a b → m b
trainOnce λ c (x, y') = state $ \l → withLearnee l train' where
	train' lt = (e, toLearnee $ over params (subtract (fromRational λ * dw)) lt) where
		(dw, e) = learnPass lt c (x, y')

trainBatch ∷ (MonadState (Learnee a b) m, Fractional b) ⇒ Rational → Cost b → [Example a b] → m b
trainBatch λ c xs = state $ \l → withLearnee l train' where
	train' lt = (e, toLearnee $ over params (subtract (fromRational λ * dw)) lt) where
		(dw, e) = (sum *** avg) ∘ unzip ∘ map (learnPass lt c) $ xs

trainEpoch ∷ (MonadState (Learnee a b) m, Fractional b) ⇒ Rational → Cost b → [[Example a b]] → m b
trainEpoch λ c = liftM (avg ∘ concat) ∘ mapM (\xs → liftM (replicate (length xs)) (trainBatch λ c xs))

instance (Monoid w, MonadRandom m) ⇒ MonadRandom (WriterT w m) where
	getRandomPrim = lift ∘ getRandomPrim

instance MonadRandom m ⇒ MonadRandom (StateT s m) where
	getRandomPrim = lift ∘ getRandomPrim

trainUntil ∷ (MonadRandom m, MonadState (Learnee a b) m, Fractional b, Ord b) ⇒ Rational → Int → Int → b → Cost b → [Example a b] → m b
trainUntil λ epochs batch eps c xs = do
	bs ← execWriterT $ train' epochs
	return $ avg bs
	where
		train' 0 = return ()
		train' n = do
			xs' ← shuffleList xs
			e ← trainEpoch λ c $ makeBatches batch xs'
			tell [e]
			unless (e < eps) $ train' (pred n)
