{-# LANGUAGE RankNTypes, FlexibleContexts, FlexibleInstances, OverloadedStrings #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Numeric.Trainee.Learnee (
	eval,
	(⇉), (‖), into, paired,
	cost, squared, logLoss, crossEntropy,
	learnee, computee,

	makeBatches, shuffleList,

	miss, avg,
	runLearnT, runLearn,
	learnPass, trainOnce, trainBatch, trainEpoch, trainUntil
	) where

import Prelude.Unicode

import Control.Arrow ((***), (>>>))
import Control.DeepSeq
import Control.Lens
import Control.Monad.State.Strict
import Control.Monad.Writer.Strict
import Data.List (unfoldr)
import Data.Random (runRVar, StdRandom(..), MonadRandom)
import Data.Random.Internal.Source
import Data.Random.List (shuffle)
import Data.Typeable (cast)
import qualified Data.Vector as V
import Numeric.AD (AD)
import Numeric.AD.Mode.Forward (Forward, auto, diff')

import Numeric.Trainee.Types

eval ∷ Learnee a b → a → b
eval (Learnee ws f) = fst ∘ f ws

(⇉) ∷ Learnee a b → Learnee b c → Learnee a c
(⇉) = (>>>)

into ∷ Learnee a b → Learnee b c → Learnee a c
into = (>>>)

(‖) ∷ Learnee a b → Learnee a' b' → Learnee (a, a') (b, b')
Learnee lws f ‖ Learnee rws g = lws `deepseq` rws `deepseq` Learnee (Params (lws, rws)) h where
	h ws (x, y) = case cast ws of
		(Just (lws', rws')) → x `seq` y `seq` x' `seq` y' `seq` lws' `deepseq` rws' `deepseq` ((x', y'), up) where
			(x', f') = f lws' x
			(y', g') = g rws' y
			up (dx', dy') = dx' `seq` dy' `seq` dx `seq` dy `seq` lws'' `deepseq` rws'' `deepseq` (Params (lws'', rws''), (dx, dy)) where
				(lws'', dx) = f' dx'
				(rws'', dy) = g' dy'
		_ → error "learnee: (||): impossible"

paired ∷ Learnee a b → Learnee a' b' → Learnee (a, a') (b, b')
paired = (‖)

cost ∷ Num a ⇒ (forall s . AD s (Forward a) → AD s (Forward a) → AD s (Forward a)) → Cost a
cost fn y' = diff' (fn (auto y'))

squared ∷ Floating a ⇒ Cost a
squared = cost $ \y' y → (y - y') ** 2

logLoss ∷ Floating a ⇒ Cost a
logLoss = cost $ \y' y → - (y' * log y)

crossEntropy ∷ Floating a ⇒ Cost a
crossEntropy = cost $ \y' y → - (y' * log y + (1 - y') * log (1 - y))

learnee ∷ Parametric w ⇒ Gradee (w, a) b → w → Learnee a b
learnee g ws = Learnee (Params ws) h where
	h ps x = case cast ps of
		Just ws' → x `seq` ws' `deepseq` y `seq` (y, back) where
			y = view (runGradee g) (ws', x)
			back dy = dy `seq` dx `seq` dws `deepseq` (Params dws, dx) where
				(dws, dx) = set (runGradee g) dy (ws', x)
		_ → error "learnee: learnee: impossible"

computee ∷ Gradee a b → Learnee a b
computee g = Learnee (Params NoParams) h where
	h _ x = x `seq` y `seq` (y, back) where
		y = view (runGradee g) x
		back dy = dy `seq` dx `seq` (Params NoParams, dx) where
			dx = set (runGradee g) dy x

makeBatches ∷ Int → [a] → [[a]]
makeBatches sz = takeWhile (not ∘ null) ∘ unfoldr (Just ∘ splitAt sz)

shuffleList ∷ MonadRandom m ⇒ [a] → m [a]
shuffleList ls = runRVar (shuffle ls) StdRandom

miss ∷ HasNorm b ⇒ Learnee a b → Cost b → Sample a b → Norm b
miss l c s = snd $ learnPass l c s

avg ∷ (Foldable t, Fractional a) ⇒ t a → a
avg ls = sum ls / fromIntegral (length ls)

runLearnT ∷ Learnee a b → StateT (Learnee a b) m c → m (c, Learnee a b)
runLearnT = flip runStateT

runLearn ∷ Learnee a b → State (Learnee a b) c → (c, Learnee a b)
runLearn = flip runState

learnPass ∷ HasNorm b ⇒ Learnee a b → Cost b → Sample a b → (Params, Norm b)
learnPass (Learnee ws f) c (Sample x y') = x `seq` ws `deepseq` y' `seq` y `seq` dws `deepseq` (dws, norm e) where
	(y, back) = f ws x
	(e, de) = c y' y
	(dws, _) = back de

trainOnce ∷ (MonadState (Learnee a b) m, HasNorm b) ⇒ Rational → Cost b → Sample a b → m (Norm b)
trainOnce λ c s = state train' where
	train' l = dw `deepseq` (e, over params (subtract (fromRational λ * dw)) l) where
		(dw, e) = learnPass l c s

trainBatch ∷ (MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b)) ⇒ Rational → Cost b → Samples a b → m (Norm b)
trainBatch λ c xs = state train' where
	train' l = dw `deepseq` (e, over params (subtract (fromRational λ * dw)) l) where
		(dw, e) = (V.sum *** avg) ∘ V.unzip ∘ V.map (learnPass l c) $ xs

trainEpoch ∷ (MonadRandom m, MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b)) ⇒ Rational → Int → Cost b → Samples a b → m (Norm b)
trainEpoch λ batch c xs = do
	ix' ← shuffleList [0 .. V.length xs - 1]
	fmap (avg ∘ concat) ∘ mapM (\ixs → fmap (replicate (length ixs)) (trainBatch λ c (samples' ixs xs))) $ makeBatches batch ix'
	where
		samples' is vs = samples $ map (vs V.!) is

instance (Monoid w, MonadRandom m) ⇒ MonadRandom (WriterT w m) where
	getRandomPrim = lift ∘ getRandomPrim

instance MonadRandom m ⇒ MonadRandom (StateT s m) where
	getRandomPrim = lift ∘ getRandomPrim

trainUntil ∷ (MonadRandom m, MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b), Ord (Norm b)) ⇒ Rational → Int → Int → Norm b → Cost b → Samples a b → m (Norm b)
trainUntil λ epochs batch eps c xs = do
	bs ← execWriterT $ train' epochs
	return $ last bs
	where
		train' 0 = return ()
		train' n = do
			e ← trainEpoch λ batch c xs
			tell [e]
			unless (e < eps) $ train' (pred n)
